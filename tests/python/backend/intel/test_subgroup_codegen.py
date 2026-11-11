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
"""OpenCL C generated for an Intel target uses sub-group builtins.

These check the generated source rather than a device result, so they also run
on machines without an Intel GPU.  The counterpart assertions against the stock
``opencl`` target guard the OpenCL backends that must keep their existing
shared-memory lowering.
"""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

NTHD = 16


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    # Source generation needs no OpenCL device.
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")


def _reduction_module(nthd: int = NTHD):
    """A cross-thread reduction, the construct that lowers to warp shuffles."""
    rows = 8

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((rows, nthd), "float32"), B: T.Tensor((rows,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", rows) as i:
                with T.launch_thread("threadIdx.x", nthd) as k:
                    partial = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                    reduced = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                    partial[0] = A[i, k]
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                        (T.float32(0),),
                        (partial[0],),
                        True,
                        (reduced[0],),
                        (k,),
                    )
                    if k == 0:
                        B[i] = reduced[0]

    return Module


def _opencl_source(target, nthd: int = NTHD):
    executable = tvm_tirx.compile(_reduction_module(nthd), target=target)
    return executable.mod.imports[0].inspect_source("cl")


def test_intel_reduction_uses_sub_group_shuffle():
    """The reduction must become a shuffle tree, not a __local memory tree."""
    source = _opencl_source("intel/arc-b60-opencl")
    assert "intel_sub_group_shuffle_down" in source
    # A shuffle tree keeps the partial sums in registers, so no __local staging
    # buffer is needed for a reduction that fits in one sub-group.
    assert "__local" not in source


def test_intel_kernel_pins_sub_group_size():
    """Without the attribute IGC may pick another SIMD width than the shuffles
    were lowered for, which would reduce across the wrong lanes."""
    source = _opencl_source("intel/arc-b60-opencl")
    assert f"intel_reqd_sub_group_size({NTHD})" in source


def test_intel_kernel_enables_sub_group_extension():
    source = _opencl_source("intel/arc-b60-opencl")
    assert "cl_intel_subgroups" in source


def test_intel_shuffle_reduction_needs_no_barrier():
    """A reduction inside one sub-group is pure data dependencies."""
    source = _opencl_source("intel/arc-b60-opencl")
    assert "barrier" not in source
    assert "mem_fence" not in source


def test_intel_warp_sync_is_a_fence_not_a_barrier():
    """A width of 100 is neither a divisor nor a multiple of the sub-group size,
    so this reduction keeps the shared-memory tree, whose in-warp tail emits
    warp-scoped synchronization from inside divergent control flow.

    A barrier of any scope is undefined in that position -- OpenCL requires every
    work-item of the work-group (``barrier``) or of the sub-group
    (``sub_group_barrier``) to reach it.  Only a fence is well defined there.
    """
    source = _opencl_source("intel/arc-b60-opencl", nthd=100)
    assert "mem_fence(CLK_LOCAL_MEM_FENCE)" in source
    assert "sub_group_barrier" not in source
    # The work-group barriers between tree levels are reached by every work-item
    # and stay as they are.
    assert "barrier(CLK_LOCAL_MEM_FENCE)" in source


def test_stock_opencl_target_is_unchanged():
    """Regression guard for every other OpenCL vendor: no sub-group builtins,
    which do not exist outside Intel implementations."""
    source = _opencl_source("opencl")
    assert "sub_group" not in source
    assert "intel_reqd_sub_group_size" not in source
    # The shared-memory reduction it has always used.
    assert "__local" in source


def test_sub_group_size_follows_the_target():
    """The pinned width comes from the target, not a hardcoded constant."""
    source = _opencl_source("intel/xe-opencl")
    assert f"intel_reqd_sub_group_size({NTHD})" in source


if __name__ == "__main__":
    tvm_tirx.testing.main()
