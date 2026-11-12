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
"""Vector literals in generated OpenCL C use OpenCL's cast syntax.

OpenCL spells a vector literal ``(int2)(x, y)``.  The C base printed ``int2(x, y)``,
CUDA's constructor syntax, which OpenCL C rejects, so any Shuffle that builds a
vector failed to compile.
"""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _module():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 1):
                    # Swap the two elements through a vector built from scalars.
                    B[T.Ramp(0, 1, 2)] = T.Shuffle([A[1], A[0]], [0, 1])

    return Module


def test_shuffle_prints_an_opencl_vector_literal(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    source = tvm_tirx.compile(_module(), target="opencl").mod.imports[0].inspect_source("cl")
    assert "(int2)(" in source
    assert " int2(" not in source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_shuffle_compiles_and_runs():
    fun = tvm_tirx.compile(_module(), target="opencl")

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        a = tvm_tirx.runtime.tensor(np.array([3, 7], np.int32), dev)
        b = tvm_tirx.runtime.tensor(np.zeros(2, np.int32), dev)
        fun["main"](a, b)
        np.testing.assert_array_equal(b.numpy(), [7, 3])

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
