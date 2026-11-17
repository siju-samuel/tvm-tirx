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
"""``T.intel.tile.ewise``: a scalar function over sub-group register fragments."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.intel.target_tags import device_tag
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout

# The tag of the Intel GPU these run on, so each tagged part is tested with its own limits.
TARGET = device_tag() or "intel/arc-b60-opencl"
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"


def lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


@T.function(private=True)
def scale_bias_relu(acc: T.float32, bias: T.float32, alpha: T.float32) -> T.float32:
    value = acc * alpha + bias
    return T.max(value, T.float32(0))


@T.function(private=True)
def plus_one(x: T.float32) -> T.float32:
    return x + T.float32(1)


@pytest.fixture
def opencl_source(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(module):
        return tvm_tirx.compile(module, target=TARGET).mod.imports[0].inspect_source("cl")

    return generate


def _single(src_layout, scope="subgroup", divergent=False):
    """dst and one fragment input over an 8 x 16 tile; the input's layout varies."""

    @I.ir_module
    class Module:
        @T.function
        def main(D: T.Tensor((8, 16), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    dr = T.alloc_local((8,), "float32")
                    sr = T.alloc_local((8,), "float32")
                    for r in range(8):
                        sr[r] = T.Cast("float32", lane)
                    if divergent and lane < 4:
                        T.intel.tile.ewise(
                            plus_one,
                            dr.view(1, 1, 8, 16, layout=lay((1, 1, 8, 16), (8, 8, 1))),
                            sr.view(1, 1, 8, 16, layout=src_layout),
                            scope=scope,
                        )
                    else:
                        T.intel.tile.ewise(
                            plus_one,
                            dr.view(1, 1, 8, 16, layout=lay((1, 1, 8, 16), (8, 8, 1))),
                            sr.view(1, 1, 8, 16, layout=src_layout),
                            scope=scope,
                        )
                    for r in range(8):
                        D[r, lane] = dr[r]

    return Module


SAME = lay((1, 1, 8, 16), (8, 8, 1))


def test_lowers_to_one_statement_per_register(opencl_source):
    source = opencl_source(_single(SAME))
    # Eight destination registers, each computed once by every work-item.
    assert source.count("+ 1.000000e+00f") == 8


def test_input_on_another_work_item_is_rejected(opencl_source):
    """Rows on sglid and columns in registers: the input holds each coordinate on a
    different work-item than dst, and ewise moves no data between them."""
    transposed = TileLayout(S[(1, 1, 8, 16) : (8, 8, 1 @ sglid, 1)])
    with pytest.raises(Exception, match="does not move data between work-items"):
        opencl_source(_single(transposed))


def test_divergent_call_needs_thread_scope(opencl_source):
    """No work-item reads another's registers, so a divergent call is safe at thread
    scope; at sub-group scope it is held to the collectives' convergence proof."""
    with pytest.raises(Exception, match="cannot prove that all 16 work-items"):
        opencl_source(_single(SAME, scope="subgroup", divergent=True))
    opencl_source(_single(SAME, scope="thread", divergent=True))


@pytest.mark.gpu
@requires_intel_gpu
def test_gemm_epilogue():
    """relu(alpha * (A @ B) + bias[n]) applied in registers before the block write.

    The bias is one value per output column.  Each work-item holds the bias of its
    column in each 16-column block, and a view with stride 0 on the row axes
    presents that as the whole tile."""
    M = N = K = 256
    MR, NR, SG_M, SG_N = 4, 2, 4, 4
    TM, TN = MR * 8, NR * 16
    WM, WN = SG_M * TM, SG_N * TN
    d = lay((MR, NR, 8, 16), (NR * 8, 8, 1))
    bias_view = lay((MR, NR, 8, 16), (0, 1, 0))
    alpha = 0.5

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            Bias: T.Tensor((N,), "float32"),
            C: T.Tensor((M, N), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.y", M // WM) as by:
                with T.launch_thread("blockIdx.x", N // WN) as bx:
                    with T.launch_thread("threadIdx.x", SG_M * SG_N * 16) as tx:
                        ar = T.alloc_local((MR * 8,), "float16")
                        br = T.alloc_local((NR * 16,), "float16")
                        dr = T.alloc_local((MR * NR * 8,), "float32")
                        bias = T.alloc_local((NR,), "float32")
                        lane = tx % 16
                        row0 = by * WM + (tx // 16 // SG_N) * TM
                        col0 = bx * WN + (tx // 16 % SG_N) * TN
                        for e in range(MR * NR * 8):
                            dr[e] = T.float32(0)
                        for j in range(NR):
                            bias[j] = Bias[col0 + j * 16 + lane]
                        for kb in T.serial(K // 16, unroll=False):
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_8r16x1c",
                                ar.view(MR, 1, 8, 16, layout=lay((MR, 1, 8, 16), (8, 8, 1))),
                                A[row0 : row0 + TM, kb * 16 : kb * 16 + 16],
                                scope="subgroup",
                            )
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_transform_16b_16r16x1c",
                                br.view(T.uint32.dtype).view(
                                    1, NR, 8, 16, layout=lay((1, NR, 8, 16), (NR * 8, 8, 1))
                                ),
                                B[kb * 16 : kb * 16 + 16, col0 : col0 + TN],
                                scope="subgroup",
                            )
                            T.intel.tile.dpas(
                                F16,
                                dr.view(MR, NR, 8, 16, layout=d),
                                ar.view(MR, 1, 8, 16, layout=lay((MR, 1, 8, 16), (8, 8, 1))),
                                br.view(T.uint32.dtype).view(
                                    1, NR, 8, 16, layout=lay((1, NR, 8, 16), (NR * 8, 8, 1))
                                ),
                                dr.view(MR, NR, 8, 16, layout=d),
                                scope="subgroup",
                            )
                        T.intel.tile.ewise(
                            scale_bias_relu,
                            dr.view(MR, NR, 8, 16, layout=d),
                            dr.view(MR, NR, 8, 16, layout=d),
                            bias.view(MR, NR, 8, 16, layout=bias_view),
                            T.float32(alpha),
                            scope="subgroup",
                        )
                        T.intel.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            C[row0 : row0 + TM, col0 : col0 + TN],
                            dr.view(MR, NR, 8, 16, layout=d),
                            scope="subgroup",
                        )

    fun = tvm_tirx.compile(Module, target=TARGET)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    bias = rng.standard_normal(N).astype(np.float32)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = [tvm_tirx.runtime.tensor(x, dev) for x in (a, b, bias, np.zeros((M, N), np.float32))]
        fun["main"](*args)
        want = np.maximum(alpha * (a.astype(np.float32) @ b.astype(np.float32)) + bias, 0)
        np.testing.assert_allclose(args[3].numpy(), want, rtol=1e-3, atol=1e-3)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
