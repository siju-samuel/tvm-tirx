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
"""``T.opencl.tile.block_read`` and ``block_write``: Intel 2D block I/O.

The ``gpu`` tests move one block with each verified variant and compare every
register word with the reference block computed in NumPy, so a wrong lane,
component or packing order anywhere fails.
"""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.intel.tile.block_io import VERIFIED, variant
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout

TARGET = "intel/arc-b60-opencl"
ROWS, PITCH_BYTES = 64, 256


def frag(r0, r1, rows):
    return TileLayout(S[(r0, r1, rows, 16) : (r1 * rows, rows, 1, 1 @ sglid)])


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


def _name(kind, bits, rows, cols, blocks=1):
    return f"intel_sub_group_2d_block_{kind}_{bits}b_{rows}r{cols}x{blocks}c"


VARIANTS = [
    _name(kind, bits, rows, cols, blocks)
    for (kind, bits), (row_choices, cols, block_choices) in VERIFIED.items()
    for rows in row_choices
    for blocks in block_choices
]
READS = [n for n in VARIANTS if variant(n).direction == "read"]
WRITES = [n for n in VARIANTS if variant(n).direction == "write"]


def _elem(bits):
    return f"uint{bits}"


def read_module(name, y, x):
    v = variant(name)
    cols = PITCH_BYTES * 8 // v.bits
    word = tvm_ffi.dtype(_elem(v.word_bits))

    @I.ir_module
    class Module:
        @T.function
        def main(M: T.Tensor((ROWS, cols), _elem(v.bits)), W: T.Tensor((v.words, 16), word)):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    r = T.alloc_local((v.words,), word)
                    T.opencl.tile.block_read(
                        name,
                        r.view(1, 1, v.words, 16, layout=frag(1, 1, v.words)),
                        M[y : y + v.rows, x : x + v.cols],
                        scope="subgroup",
                    )
                    for q in range(v.words):
                        W[q, lane] = r[q]

    return Module, cols


def expected_words(name, block):
    """The register words the hardware produces for ``block``, per the table."""
    v = variant(name)
    if v.kind == "read" and v.bits == 8:
        block = block.astype(np.uint16)
        return block[:, 0::2] | (block[:, 1::2] << 8)
    if v.kind == "read_transform":
        block = block.astype(np.uint32)
        per = 32 // v.bits
        return sum(block[t::per] << (v.bits * t) for t in range(per)).astype(np.uint32)
    return block


@pytest.fixture
def opencl_source(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(module, target=TARGET):
        return tvm_tirx.compile(module, target=target).mod.imports[0].inspect_source("cl")

    return generate


def test_lowers_to_the_named_builtin(opencl_source):
    name = _name("read_transform", 16, 16, 16)
    module, _ = read_module(name, 16, 32)
    source = opencl_source(module)
    assert f"{name}(M_ptr, 256, 64, 256, (int2)(" in source
    assert "intel_reqd_sub_group_size(16)" in source


def _reject(opencl_source, name, region, *, words=None, layout=None, cols=None, target=TARGET):
    v = variant(name)
    cols = cols or PITCH_BYTES * 8 // v.bits
    words = words or v.words
    word = tvm_ffi.dtype(_elem(v.word_bits))
    layout = layout or frag(1, 1, words)
    (y0, y1), (x0, x1) = region

    @T.function
    def main(M: T.Tensor((ROWS, cols), _elem(v.bits)), W: T.Tensor((words, 16), word)):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16) as lane:
                r = T.alloc_local((32,), word)
                T.opencl.tile.block_read(
                    name,
                    r.view(1, 1, words, 16, layout=layout),
                    M[y0:y1, x0:x1],
                    scope="subgroup",
                )
                for q in range(words):
                    W[q, lane] = r[q]

    opencl_source(main, target=target)


R16 = _name("read", 16, 8, 16)
REJECTIONS = {
    "unverified variant": (
        dict(name=_name("read", 16, 64, 16), region=((0, 64), (0, 16))),
        "has not been verified",
    ),
    "pitch not a multiple of 16 bytes": (
        dict(name=R16, region=((0, 8), (0, 16)), cols=66),
        "pitch must be a multiple of 16 bytes",
    ),
    "x offset not 4-byte aligned": (
        dict(name=R16, region=((0, 8), (1, 17))),
        "x offset must be 4-byte aligned",
    ),
    "region not whole blocks": (
        dict(name=R16, region=((0, 12), (0, 16))),
        "not a whole number of",
    ),
    "transposed register layout": (
        dict(
            name=R16,
            region=((0, 8), (0, 16)),
            layout=TileLayout(S[(1, 1, 8, 16) : (8, 8, 1 @ sglid, 1)]),
        ),
        "work-item index within the sub-group",
    ),
    "non-Intel target": (
        dict(name=R16, region=((0, 8), (0, 16)), target="opencl"),
        "requires an Intel OpenCL target",
    ),
}


@pytest.mark.parametrize("case", REJECTIONS, ids=list(REJECTIONS))
def test_unsafe_block_is_rejected(opencl_source, case):
    """Each of these silently returns wrong data on hardware, so it must not compile."""
    kwargs, reason = REJECTIONS[case]
    with pytest.raises(Exception, match=reason):
        _reject(opencl_source, **kwargs)


def test_write_variant_is_not_a_read(opencl_source):
    with pytest.raises(Exception, match="belongs to block_write"):
        _reject(opencl_source, _name("write", 32, 8, 16), ((0, 8), (0, 16)))


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("name", READS, ids=lambda n: n.removeprefix("intel_sub_group_2d_block_"))
def test_read_matches_reference_block(name):
    v = variant(name)
    y, x = 16, 2 * v.cols
    module, cols = read_module(name, y, x)
    fun = tvm_tirx.compile(module, target=TARGET)
    rng = np.random.default_rng(0)
    m = rng.integers(0, 2**v.bits, (ROWS, cols), dtype=np.uint64).astype(_elem(v.bits))
    want = expected_words(name, m[y : y + v.rows, x : x + v.cols])

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = [
            tvm_tirx.runtime.tensor(m, dev),
            tvm_tirx.runtime.tensor(np.zeros((v.words, 16), _elem(v.word_bits)), dev),
        ]
        fun["main"](*args)
        np.testing.assert_array_equal(args[1].numpy(), want)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_intel_gpu
def test_read_past_the_edge_returns_zero():
    """A block overhanging the matrix reads zero outside it, so edge tiles need no masking."""
    name = _name("read", 16, 8, 16)
    module, cols = read_module(name, ROWS - 4, PITCH_BYTES // 2 - 8)
    fun = tvm_tirx.compile(module, target=TARGET)
    m = np.arange(ROWS * cols, dtype=np.uint16).reshape(ROWS, cols) + 1
    want = np.zeros((8, 16), np.uint16)
    want[:4, :8] = m[ROWS - 4 :, cols - 8 :]

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = [
            tvm_tirx.runtime.tensor(m, dev),
            tvm_tirx.runtime.tensor(np.zeros((8, 16), np.uint16), dev),
        ]
        fun["main"](*args)
        np.testing.assert_array_equal(args[1].numpy(), want)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def write_module(name, y, x):
    v = variant(name)
    cols = PITCH_BYTES * 8 // v.bits
    word = tvm_ffi.dtype(_elem(v.word_bits))

    @I.ir_module
    class Module:
        @T.function
        def main(W: T.Tensor((v.words, 16), word), M: T.Tensor((ROWS, cols), _elem(v.bits))):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    r = T.alloc_local((v.words,), word)
                    for q in range(v.words):
                        r[q] = W[q, lane]
                    T.opencl.tile.block_write(
                        name,
                        M[y : y + v.rows, x : x + v.cols],
                        r.view(1, 1, v.words, 16, layout=frag(1, 1, v.words)),
                        scope="subgroup",
                    )

    return Module, cols


def _run_write(name, y, x, sentinel):
    v = variant(name)
    module, cols = write_module(name, y, x)
    fun = tvm_tirx.compile(module, target=TARGET)
    words = np.random.default_rng(1).integers(0, 2**v.bits, (v.words, 16), dtype=np.uint64)
    words = words.astype(_elem(v.word_bits))
    result = {}

    def run():
        dev = tvm_tirx.opencl(0)
        args = [
            tvm_tirx.runtime.tensor(words, dev),
            tvm_tirx.runtime.tensor(np.full((ROWS, cols), sentinel, _elem(v.bits)), dev),
        ]
        fun["main"](*args)
        result["m"] = args[1].numpy()

    tvm_tirx.testing.run_with_gpu_lock(run)
    return words, result["m"], cols


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("name", WRITES, ids=lambda n: n.removeprefix("intel_sub_group_2d_block_"))
def test_write_stores_the_reference_block(name):
    v = variant(name)
    y, x = 8, 32
    words, m, cols = _run_write(name, y, x, sentinel=7)
    want = np.full((ROWS, cols), 7, _elem(v.bits))
    want[y : y + v.rows, x : x + 16] = words
    np.testing.assert_array_equal(m, want)


@pytest.mark.gpu
@requires_intel_gpu
def test_write_past_the_edge_is_clipped():
    """Only the part of an overhanging block inside the matrix is stored; nothing wraps
    into the next row."""
    name = _name("write", 32, 8, 16)
    words, m, cols = _run_write(name, ROWS - 4, PITCH_BYTES // 4 - 8, sentinel=7)
    want = np.full((ROWS, cols), 7, np.uint32)
    want[ROWS - 4 :, cols - 8 :] = words[:4, :8]
    np.testing.assert_array_equal(m, want)


@pytest.mark.gpu
@requires_intel_gpu
def test_gemm_on_block_io():
    """Block reads feed DPAS and a block write stores the result, with the same views."""
    M = N = K = 512
    MR, NR, SG_M, SG_N = 4, 2, 4, 4
    TM, TN = MR * 8, NR * 16
    WM, WN = SG_M * TM, SG_N * TN
    F16 = "intel_sub_group_f16_f16_matrix_mad_k16"

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
                        row0 = by * WM + (tx // 16 // SG_N) * TM
                        col0 = bx * WN + (tx // 16 % SG_N) * TN
                        for e in range(MR * NR * 8):
                            dr[e] = T.float32(0)
                        for kb in range(K // 16):
                            T.opencl.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_8r16x1c",
                                ar.view(MR, 1, 8, 16, layout=frag(MR, 1, 8)),
                                A[row0 : row0 + TM, kb * 16 : kb * 16 + 16],
                                scope="subgroup",
                            )
                            T.opencl.tile.block_read(
                                "intel_sub_group_2d_block_read_transform_16b_16r16x1c",
                                br.view(T.uint32.dtype).view(1, NR, 8, 16, layout=frag(1, NR, 8)),
                                B[kb * 16 : kb * 16 + 16, col0 : col0 + TN],
                                scope="subgroup",
                            )
                            T.opencl.tile.dpas(
                                F16,
                                dr.view(MR, NR, 8, 16, layout=frag(MR, NR, 8)),
                                ar.view(MR, 1, 8, 16, layout=frag(MR, 1, 8)),
                                br.view(T.uint32.dtype).view(1, NR, 8, 16, layout=frag(1, NR, 8)),
                                dr.view(MR, NR, 8, 16, layout=frag(MR, NR, 8)),
                                scope="subgroup",
                            )
                        T.opencl.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            C[row0 : row0 + TM, col0 : col0 + TN],
                            dr.view(MR, NR, 8, 16, layout=frag(MR, NR, 8)),
                            scope="subgroup",
                        )

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


def _prefetch_function(name, region):
    (y0, y1), (x0, x1) = region

    @T.function
    def main(M: T.Tensor((ROWS, 128), "uint16")):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16):
                T.opencl.tile.block_prefetch(name, M[y0:y1, x0:x1], scope="subgroup")

    return main


def test_prefetch_lowers_to_one_call_per_block(opencl_source):
    """A 64 x 32 region is two 32 x 32 blocks of the two-block-wide variant."""
    name = _name("prefetch", 16, 32, 16, 2)
    source = opencl_source(_prefetch_function(name, ((0, 64), (32, 64))))
    assert source.count(f"{name}(M_ptr, 256, 64, 256, (int2)(") == 2


PREFETCH_REJECTIONS = {
    "read variant": (_name("read", 16, 8, 16), "belongs to block_read"),
    "unverified width": (
        _name("prefetch", 16, 8, 16).replace("x1c", "x4c"),
        "has not been verified",
    ),
}


@pytest.mark.parametrize("case", PREFETCH_REJECTIONS, ids=list(PREFETCH_REJECTIONS))
def test_prefetch_rejects(opencl_source, case):
    name, reason = PREFETCH_REJECTIONS[case]
    with pytest.raises(Exception, match=reason):
        opencl_source(_prefetch_function(name, ((0, 8), (0, 64))))


def test_reads_are_not_accepted_two_blocks_wide(opencl_source):
    """Only prefetch was checked two blocks wide; a read of that width is rejected."""
    with pytest.raises(Exception, match="has not been verified"):
        _reject(opencl_source, _name("read", 16, 8, 16, 2), ((0, 8), (0, 32)))


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


@pytest.mark.gpu
@requires_intel_gpu
def test_large_tile_gemm():
    """The large-tile GEMM: 64 x 32 sub-group tiles, a K step of 32, a one-step
    prefetch, and a K loop kept rolled.  Each register buffer is viewed twice, once
    as block-read atoms and once as DPAS fragments, so the layouts must agree.
    Without the automatic large register file this kernel spills; it stays correct
    but runs over 20x slower, so this checks the result, not the speed."""
    M = N = K = 512
    TM, TN, TK, SG_M, SG_N = 64, 32, 32, 4, 4
    WM, WN = SG_M * TM, SG_N * TN
    a_read, a_dpas = _lay((2, 2, 32, 16), (32, 64, 1)), _lay((8, 2, 8, 16), (8, 64, 1))
    b_read, b_dpas = _lay((1, 2, 16, 16), (32, 16, 1)), _lay((2, 2, 8, 16), (8, 16, 1))
    d = _lay((8, 2, 8, 16), (16, 8, 1))
    prefetch = "intel_sub_group_2d_block_prefetch_16b_32r16x2c"

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
                        ar = T.alloc_local((128,), "float16")
                        br = T.alloc_local((64,), "float16")
                        dr = T.alloc_local((128,), "float32")
                        row0 = by * WM + (tx // 16 // SG_N) * TM
                        col0 = bx * WN + (tx // 16 % SG_N) * TN
                        for e in range(128):
                            dr[e] = T.float32(0)
                        for kb in T.serial(K // TK, unroll=False):
                            k = kb * TK
                            T.opencl.tile.block_prefetch(
                                prefetch, A[row0 : row0 + TM, k + TK : k + 2 * TK], scope="subgroup"
                            )
                            T.opencl.tile.block_prefetch(
                                prefetch, B[k + TK : k + 2 * TK, col0 : col0 + TN], scope="subgroup"
                            )
                            T.opencl.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_32r16x1c",
                                ar.view(2, 2, 32, 16, layout=a_read),
                                A[row0 : row0 + TM, k : k + TK],
                                scope="subgroup",
                            )
                            T.opencl.tile.block_read(
                                "intel_sub_group_2d_block_read_transform_16b_32r16x1c",
                                br.view(T.uint32.dtype).view(1, 2, 16, 16, layout=b_read),
                                B[k : k + TK, col0 : col0 + TN],
                                scope="subgroup",
                            )
                            T.opencl.tile.dpas(
                                "intel_sub_group_f16_f16_matrix_mad_k16",
                                dr.view(8, 2, 8, 16, layout=d),
                                ar.view(8, 2, 8, 16, layout=a_dpas),
                                br.view(T.uint32.dtype).view(2, 2, 8, 16, layout=b_dpas),
                                dr.view(8, 2, 8, 16, layout=d),
                                scope="subgroup",
                            )
                        T.opencl.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            C[row0 : row0 + TM, col0 : col0 + TN],
                            dr.view(8, 2, 8, 16, layout=d),
                            scope="subgroup",
                        )

    fun = tvm_tirx.compile(Module, target=TARGET)
    source = fun.mod.imports[0].inspect_source("cl")
    assert "// Build options: -cl-intel-enable-auto-large-GRF-mode" in source
    assert "#pragma unroll 1" in source
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
