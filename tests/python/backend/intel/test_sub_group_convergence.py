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
"""Every work-item of a sub-group must reach a sub-group tile instruction.

A sub-group instruction is a collective, undefined in OpenCL unless the whole
sub-group reaches it.  TileDispatch proves that before lowering one; these check
that it accepts each way of staying converged and rejects each way of diverging,
naming the cause.  They need no device.
"""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import S, TileLayout

TARGET = "intel/arc-b60-opencl"
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"
FRAG = TileLayout(S[(1, 1, 8, 16) : (8, 8, 1, 1 @ sglid)])


@T.macro
def mma(ar, br, dr):
    T.opencl.tile.dpas(
        F16,
        dr.view(1, 1, 8, 16, layout=FRAG),
        ar.view(1, 1, 8, 16, layout=FRAG),
        br.view(T.uint32.dtype).view(1, 1, 8, 16, layout=FRAG),
        T.float32(0),
        scope="subgroup",
    )


def _compile(module):
    return tvm_tirx.compile(module, target=TARGET)


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")


# ---------------------------------------------------------------- converged


def test_whole_work_group_of_sub_groups():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 4):
                with T.launch_thread("threadIdx.x", 256):
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    mma(ar, br, dr)

    _compile(Module)


def test_condition_on_sub_group_id_is_uniform():
    """tx // 16 is the same for every work-item of a sub-group."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 64) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    sg = tx // 16
                    if sg < 2:
                        mma(ar, br, dr)

    _compile(Module)


def test_condition_on_work_group_and_loop_is_uniform():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 8) as bx:
                with T.launch_thread("threadIdx.x", 32):
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    for k in range(bx + 1):
                        if bx % 2 == 0:
                            mma(ar, br, dr)
                        else:
                            mma(ar, br, dr)

    _compile(Module)


def test_global_load_at_uniform_address_is_uniform():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "int32")):
            with T.launch_thread("blockIdx.x", 8) as bx:
                with T.launch_thread("threadIdx.x", 16):
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    if A[bx] > 0:
                        mma(ar, br, dr)

    _compile(Module)


# ---------------------------------------------------------------- divergent


def _rejected(module, reason):
    with pytest.raises(Exception, match=reason):
        _compile(module)


def test_partial_sub_group():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 24):
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    mma(ar, br, dr)

    _rejected(Module, "x extent, 24, is not a multiple of the sub-group size 16")


def test_condition_on_lane():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 32) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    if tx % 16 < 8:
                        mma(ar, br, dr)

    _rejected(Module, "inside `if .*whose condition can differ")


def test_else_branch_of_condition_on_lane():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    if tx < 8:
                        A[0] = T.float32(1)
                    else:
                        mma(ar, br, dr)

    _rejected(Module, "in the else branch of")


def test_condition_through_a_let_bound_lane():
    """The check sees through a let binding to the lane it hides."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    lane = tx % 16
                    if lane == 0:
                        mma(ar, br, dr)

    _rejected(Module, "whose condition can differ")


def test_condition_on_private_value():
    """Each work-item reads its own register, even at the same index."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16):
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    if dr[0] > T.float32(0):
                        mma(ar, br, dr)

    _rejected(Module, "whose condition can differ")


def test_loop_bounds_on_lane():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    for k in range(tx):
                        mma(ar, br, dr)

    _rejected(Module, "has bounds that can differ")


def test_divergent_break_in_enclosing_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    for k in range(4):
                        mma(ar, br, dr)
                        if tx == k:
                            break

    _rejected(Module, "contains a break, continue or return")


def test_divergent_return_before_the_instruction():
    # A TIRx return carries a value; the kernel never uses it, the check only needs
    # the early exit.
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "float32")):
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as tx:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    dr = T.alloc_local((8,), "float32")
                    if tx >= 8:
                        return T.int32(0)
                    mma(ar, br, dr)

    _rejected(Module, "an earlier return")


if __name__ == "__main__":
    tvm_tirx.testing.main()
