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
"""``T.intel.tile.reduce``: reductions over sub-group register fragments."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.intel.target_tags import device_tag
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import R, S, TileLayout

# The tag of the Intel GPU these run on, so each tagged part is tested with its own limits.
TARGET = device_tag() or "intel/arc-b60-opencl"
NR = 2  # 16-column blocks: an 8 x 32 tile laid out like a DPAS accumulator


def lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


# Rows in registers, columns across the sub-group: the DPAS accumulator layout.
ACC = lay((1, NR, 8, 16), (NR * 8, 8, 1))
# One value per row, held by every work-item: what a reduction over the columns leaves.
ROWS = TileLayout(S[(1, 1, 8, 1) : (8, 8, 1, 1)] + R[16 : 1 @ sglid])
# The same rows presented as the whole tile, for ewise: stride 0 along the columns.
ROWS_AS_TILE = TileLayout(S[(1, NR, 8, 16) : (0, 0, 1, 0)] + R[16 : 1 @ sglid])


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


@T.function(private=True)
def exp_minus(x: T.float32, m: T.float32) -> T.float32:
    return T.exp(x - m)


@T.function(private=True)
def divide(x: T.float32, d: T.float32) -> T.float32:
    return x / d


def _softmax_rows():
    """Softmax over the 32 columns of each of 8 rows, all in one sub-group's registers."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8, NR * 16), "float32"), B: T.Tensor((8, NR * 16), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    s = T.alloc_local((NR * 8,), "float32")
                    m = T.alloc_local((8,), "float32")
                    total = T.alloc_local((8,), "float32")
                    for j, r in T.grid(NR, 8):
                        s[j * 8 + r] = A[r, j * 16 + lane]
                    T.intel.tile.reduce(
                        "max",
                        m.view(1, 1, 8, 1, layout=ROWS),
                        s.view(1, NR, 8, 16, layout=ACC),
                    )
                    T.intel.tile.ewise(
                        exp_minus,
                        s.view(1, NR, 8, 16, layout=ACC),
                        s.view(1, NR, 8, 16, layout=ACC),
                        m.view(1, NR, 8, 16, layout=ROWS_AS_TILE),
                    )
                    T.intel.tile.reduce(
                        "sum",
                        total.view(1, 1, 8, 1, layout=ROWS),
                        s.view(1, NR, 8, 16, layout=ACC),
                    )
                    T.intel.tile.ewise(
                        divide,
                        s.view(1, NR, 8, 16, layout=ACC),
                        s.view(1, NR, 8, 16, layout=ACC),
                        total.view(1, NR, 8, 16, layout=ROWS_AS_TILE),
                    )
                    for j, r in T.grid(NR, 8):
                        B[r, j * 16 + lane] = s[j * 8 + r]

    return Module


def _single(dst_layout, dst_shape, scope="subgroup", dtype="float32"):
    """One reduce of the 8 x 32 accumulator into ``dst_shape``."""
    count = int(np.prod(dst_shape))

    @I.ir_module
    class Module:
        @T.function
        def main(D: T.Tensor((16, count), dtype)):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    s = T.alloc_local((NR * 8,), "float32")
                    d = T.alloc_local((count,), dtype)
                    for e in range(NR * 8):
                        s[e] = T.Cast("float32", lane + e)
                    T.intel.tile.reduce(
                        "sum",
                        d.view(*dst_shape, layout=dst_layout),
                        s.view(1, NR, 8, 16, layout=ACC),
                        scope=scope,
                    )
                    for e in range(count):
                        D[lane, e] = d[e]

    return Module


@pytest.fixture
def opencl_source(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(module):
        return tvm_tirx.compile(module, target=TARGET).mod.imports[0].inspect_source("cl")

    return generate


def test_reduction_over_the_sub_group_uses_the_builtin(opencl_source):
    """Each row combines its two column blocks in registers, then across the sub-group."""
    source = opencl_source(_single(ROWS, (1, 1, 8, 1)))
    assert source.count("sub_group_reduce_add(") == 8


def test_reduction_within_a_work_item_moves_no_data(opencl_source):
    """Summing the two column blocks keeps each column on its work-item."""
    source = opencl_source(_single(lay((1, 1, 8, 16), (8, 8, 1)), (1, 1, 8, 16)))
    assert "sub_group_reduce" not in source


def test_result_of_a_sub_group_reduction_needs_every_work_item(opencl_source):
    """Rows reduced across the sub-group cannot stay on one work-item per column."""
    with pytest.raises(Exception, match="must replicate it there"):
        opencl_source(_single(lay((1, 1, 8, 1), (8, 8, 1)), (1, 1, 8, 1)))


def test_replica_must_cover_the_sub_group(opencl_source):
    half = TileLayout(S[(1, 1, 8, 1) : (8, 8, 1, 1)] + R[8 : 1 @ sglid])
    with pytest.raises(Exception, match="must cover the whole sub-group"):
        opencl_source(_single(half, (1, 1, 8, 1)))


def test_reduction_across_work_items_needs_sub_group_scope(opencl_source):
    with pytest.raises(Exception, match='needs scope="subgroup"'):
        opencl_source(_single(ROWS, (1, 1, 8, 1), scope="thread"))


def test_dtypes_must_match(opencl_source):
    with pytest.raises(Exception, match="dst is float16 but src is float32"):
        opencl_source(_single(ROWS, (1, 1, 8, 1), dtype="float16"))


@pytest.mark.gpu
@requires_intel_gpu
def test_softmax_over_rows():
    fun = tvm_tirx.compile(_softmax_rows(), target=TARGET)
    a = np.random.default_rng(0).standard_normal((8, NR * 16)).astype(np.float32) * 4

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        args = [tvm_tirx.runtime.tensor(x, dev) for x in (a, np.zeros_like(a))]
        fun["main"](*args)
        e = np.exp(a - a.max(axis=1, keepdims=True))
        np.testing.assert_allclose(args[1].numpy(), e / e.sum(axis=1, keepdims=True), rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
