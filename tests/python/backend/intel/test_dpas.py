# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""``T.opencl.tile.dpas``: Intel Xe matrix multiply-accumulate on sub-group fragments.

The codegen and verifier tests need no device.  The ``gpu`` tests run each
builtin on an Intel GPU and require bit-exact agreement with a reference
product, using small integers so every intermediate is exact in the inputs'
precision; a fragment mapping that is wrong anywhere changes the result.
"""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout

TARGET = "intel/arc-b60-opencl"
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"


def frag(r0, r1, rows):
    """Every dpas atom puts (row, col) on work-item col, at register component row."""
    return TileLayout(S[(r0, r1, rows, 16) : (r1 * rows, rows, 1, 1 @ sglid)])


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


def matmul_module(instruction, elem, acc, kelems, mr, nr, kr, m=8):
    """D = A @ B + C for one sub-group, loading fragments by the documented mapping."""
    a_words = kelems // 16  # K elements per 16-bit A word
    a_word = tvm_ffi.dtype("int16" if a_words == 2 else elem)
    M, N, K = mr * m, nr * 16, kr * kelems

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((M, K), elem),
            B: T.Tensor((K, N), elem),
            C: T.Tensor((M, N), acc),
            D: T.Tensor((M, N), acc),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    ar = T.alloc_local((mr * kr * m * a_words,), elem)
                    br = T.alloc_local((kr * nr * kelems,), elem)
                    dr = T.alloc_local((mr * nr * m,), acc)
                    for i, k, r, w in T.grid(mr, kr, m, a_words):
                        ar[((i * kr + k) * m + r) * a_words + w] = A[
                            i * m + r, k * kelems + lane * a_words + w
                        ]
                    for k, j, e in T.grid(kr, nr, kelems):
                        br[(k * nr + j) * kelems + e] = B[k * kelems + e, j * 16 + lane]
                    for i, j, r in T.grid(mr, nr, m):
                        dr[(i * nr + j) * m + r] = C[i * m + r, j * 16 + lane]
                    T.opencl.tile.dpas(
                        instruction,
                        dr.view(mr, nr, m, 16, layout=frag(mr, nr, m)),
                        ar.view(a_word).view(mr, kr, m, 16, layout=frag(mr, kr, m)),
                        br.view(T.uint32.dtype).view(kr, nr, 8, 16, layout=frag(kr, nr, 8)),
                        dr.view(mr, nr, m, 16, layout=frag(mr, nr, m)),
                        scope="subgroup",
                    )
                    for i, j, r in T.grid(mr, nr, m):
                        D[i * m + r, j * 16 + lane] = dr[(i * nr + j) * m + r]

    return Module, (M, N, K)


def single_module(a_layout=None, d_layout=None, b_view=True, instruction=F16, acc="float32"):
    """One f16 dpas whose operand layouts a test can replace."""
    a_layout = a_layout or frag(1, 1, 8)
    d_layout = d_layout or frag(1, 1, 8)
    # Without the uint32 view, B arrives as 16-bit halves rather than packed words.
    b_word = tvm_ffi.dtype("uint32" if b_view else "float16")

    @T.function
    def main(D: T.Tensor((8, 16), "float32")):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16) as lane:
                ar = T.alloc_local((16,), "float16")
                br = T.alloc_local((16,), "float16")
                dr = T.alloc_local((16,), acc)
                T.opencl.tile.dpas(
                    instruction,
                    dr.view(1, 1, 8, 16, layout=d_layout),
                    ar.view(1, 1, 8, 16, layout=a_layout),
                    br.view(b_word).view(1, 1, 8, 16, layout=frag(1, 1, 8)),
                    T.float32(0),
                    scope="subgroup",
                )
                for r in range(8):
                    D[r, lane] = T.Cast("float32", dr[r])

    return main


@pytest.fixture
def opencl_source(monkeypatch):
    # Source generation needs no OpenCL device.
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(module, target=TARGET):
        return tvm_tirx.compile(module, target=target).mod.imports[0].inspect_source("cl")

    return generate


def test_lowers_to_the_named_builtin(opencl_source):
    module, _ = matmul_module(F16, "float16", "float32", 16, 2, 2, 2)
    source = opencl_source(module)
    # 2 x 2 output fragments, each accumulating over 2 K steps.
    assert source.count(f"{F16}(") == 8
    assert "intel_reqd_sub_group_size(16)" in source
    assert "cl_intel_subgroups" in source


def test_unsigned_families_take_unsigned_vectors(opencl_source):
    """The unsigned builtins have no signed overloads, and OpenCL vectors do not
    convert implicitly, so the operands are reinterpreted to exactly its types."""
    module, _ = matmul_module("intel_sub_group_u8_i8_matrix_mad_k32", "uint8", "int32", 32, 1, 1, 1)
    source = opencl_source(module)
    assert "ushort8" in source


REJECTIONS = {
    "transposed atom": (
        dict(a_layout=TileLayout(S[(1, 1, 8, 16) : (8, 8, 1 @ sglid, 1)])),
        "work-item index within the sub-group",
    ),
    "strided components": (
        dict(d_layout=TileLayout(S[(1, 1, 8, 16) : (16, 16, 2, 1 @ sglid)])),
        "consecutive registers",
    ),
    "B not packed into 32-bit words": (dict(b_view=False), "expected scalar 32-bit words"),
    "unknown instruction": (
        dict(instruction="intel_sub_group_f32_mad"),
        "Unknown dpas instruction",
    ),
    "wrong accumulator type": (dict(acc="int32"), "accumulates in float32 or float16"),
}


@pytest.mark.parametrize("case", REJECTIONS, ids=list(REJECTIONS))
def test_mismatched_fragment_is_rejected(opencl_source, case):
    """A fragment that disagrees with the hardware fails to compile with its reason,
    rather than returning plausible but wrong numbers."""
    kwargs, reason = REJECTIONS[case]
    with pytest.raises(Exception, match=reason):
        opencl_source(single_module(**kwargs))


def test_accumulator_repeats_must_not_alias(opencl_source):
    """Two output repeats in the same registers would overwrite each other."""
    aliased = TileLayout(S[(2, 1, 8, 16) : (0, 8, 1, 1 @ sglid)])

    @T.function
    def main(D: T.Tensor((8, 16), "float32")):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16) as lane:
                ar = T.alloc_local((16,), "float16")
                br = T.alloc_local((16,), "float16")
                dr = T.alloc_local((8,), "float32")
                T.opencl.tile.dpas(
                    F16,
                    dr.view(2, 1, 8, 16, layout=aliased),
                    ar.view(2, 1, 8, 16, layout=frag(2, 1, 8)),
                    br.view(T.uint32.dtype).view(1, 1, 8, 16, layout=frag(1, 1, 8)),
                    T.float32(0),
                    scope="subgroup",
                )
                for r in range(8):
                    D[r, lane] = dr[r]

    with pytest.raises(Exception, match="share registers"):
        opencl_source(main)


def test_print_parse_round_trip():
    """The printer spells the scope as a ThreadScope object; the parser must accept it."""
    from tvm_tirx.ir.base import assert_structural_equal

    function = single_module()
    text = function.script()
    assert "T.opencl.tile.dpas(" in text
    reparsed = tvm_tirx.script.from_source(text, extra_vars={"T": T, "I": I})
    assert_structural_equal(function, reparsed)


def test_requires_an_intel_target(opencl_source):
    with pytest.raises(Exception, match="requires an Intel OpenCL target"):
        opencl_source(single_module(), target="opencl")


FAMILIES = [
    (F16, "float16", "float32", 16),
    (F16, "float16", "float16", 16),
    ("intel_sub_group_bf16_bf16_matrix_mad_k16", "bfloat16", "float32", 16),
    ("intel_sub_group_i8_i8_matrix_mad_k32", "int8", "int32", 32),
    ("intel_sub_group_u8_u8_matrix_mad_k32", "uint8", "int32", 32),
    ("intel_sub_group_u8_i8_matrix_mad_k32", "uint8", "int32", 32),
    ("intel_sub_group_i8_u8_matrix_mad_k32", "uint8", "int32", 32),
]
SHAPES = [(1, 1, 1, 8), (2, 2, 2, 8), (3, 2, 4, 8), (2, 3, 2, 4), (2, 1, 3, 1)]


def _operands(instruction, elem, acc, M, N, K, rng):
    if elem in ("float16", "bfloat16"):
        a = rng.integers(-3, 4, (M, K)).astype(np.float32)
        b = rng.integers(-3, 4, (K, N)).astype(np.float32)
        c = rng.integers(-3, 4, (M, N)).astype(np.float32)
        want = a @ b + c
        if elem == "float16":
            return a.astype(np.float16), b.astype(np.float16), c.astype(acc), want.astype(acc)
        # bfloat16 is the high half of a float32; small integers are exact.
        a, b = ((x.view(np.uint32) >> 16).astype(np.uint16) for x in (a, b))
        return a, b, c, want
    signs = instruction.removeprefix("intel_sub_group_").split("_")[:2]

    def gen(sign, shape):
        return rng.integers(-8, 8, shape) if sign == "i8" else rng.integers(0, 16, shape)

    a, b = gen(signs[0], (M, K)), gen(signs[1], (K, N))
    c = rng.integers(-8, 8, (M, N))
    store = np.int8 if elem == "int8" else np.uint8  # two's complement bits
    return a.astype(store), b.astype(store), c.astype(np.int32), (a @ b + c).astype(np.int32)


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "family", FAMILIES, ids=lambda f: f[0].removeprefix("intel_sub_group_")[:10] + "/" + f[2]
)
def test_matches_reference_product(family, shape):
    instruction, elem, acc, kelems = family
    mr, nr, kr, m = shape
    module, (M, N, K) = matmul_module(instruction, elem, acc, kelems, mr, nr, kr, m)
    fun = tvm_tirx.compile(module, target=TARGET)
    a, b, c, want = _operands(instruction, elem, acc, M, N, K, np.random.default_rng(0))

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = []
        for x, dtype in ((a, elem), (b, elem)):
            t = tvm_tirx.runtime.empty(x.shape, dtype, dev)
            t.copyfrom(x)
            args.append(t)
        args.append(tvm_tirx.runtime.tensor(c, dev))
        out = tvm_tirx.runtime.tensor(np.zeros((M, N), want.dtype), dev)
        fun["main"](*args, out)
        np.testing.assert_array_equal(out.numpy(), want)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_intel_gpu
def test_gemm_across_work_groups():
    """A tiled GEMM: several sub-groups per work-group, many work-groups, and a K
    loop, which also exercises the sub-group-to-work-item mapping the fragment
    loads assume (lane = local id % 16)."""
    M = N = K = 256
    MR, NR, SG_M, SG_N = 4, 2, 2, 2
    TM, TN = MR * 8, NR * 16
    WM, WN = SG_M * TM, SG_N * TN

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            C: T.Tensor((M, N), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.y", M // WM) as by:
                with T.launch_thread("blockIdx.x", N // WN) as bx:
                    with T.launch_thread("threadIdx.x", SG_M * SG_N * 16) as tx:
                        ar = T.alloc_local((MR * 8,), "float16")
                        br = T.alloc_local((NR * 16,), "float16")
                        dr = T.alloc_local((MR * NR * 8,), "float32")
                        lane = tx % 16
                        row0 = by * WM + (tx // 16 // SG_N) * TM
                        col0 = bx * WN + (tx // 16 % SG_N) * TN
                        for e in range(MR * NR * 8):
                            dr[e] = T.float32(0)
                        for kb in range(K // 16):
                            for i, r in T.grid(MR, 8):
                                ar[i * 8 + r] = A[row0 + i * 8 + r, kb * 16 + lane]
                            for j, e in T.grid(NR, 16):
                                br[j * 16 + e] = B[kb * 16 + e, col0 + j * 16 + lane]
                            T.opencl.tile.dpas(
                                F16,
                                dr.view(MR, NR, 8, 16, layout=frag(MR, NR, 8)),
                                ar.view(MR, 1, 8, 16, layout=frag(MR, 1, 8)),
                                br.view(T.uint32.dtype).view(1, NR, 8, 16, layout=frag(1, NR, 8)),
                                dr.view(MR, NR, 8, 16, layout=frag(MR, NR, 8)),
                                scope="subgroup",
                            )
                        for i, j, r in T.grid(MR, NR, 8):
                            C[row0 + i * 8 + r, col0 + j * 16 + lane] = dr[(i * NR + j) * 8 + r]

    fun = tvm_tirx.compile(Module, target=TARGET)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = [tvm_tirx.runtime.tensor(x, dev) for x in (a, b, np.zeros((M, N), np.float32))]
        fun["main"](*args)
        want = a.astype(np.float32) @ b.astype(np.float32)
        np.testing.assert_allclose(args[2].numpy(), want, rtol=1e-3, atol=1e-3)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
