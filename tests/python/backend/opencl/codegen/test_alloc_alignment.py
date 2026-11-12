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
"""Stack arrays in generated OpenCL C keep the alignment the IR records for them.

OpenCL aligns a variable only to its own element type, while views of a
tensor's storage rely on its declared data_alignment -- a uint32 view of a half
array is read through a uint pointer, which OpenCL defines only for 4-byte
aligned storage.  These need no OpenCL device.
"""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.fixture
def opencl_source(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(module):
        return tvm_tirx.compile(module, target="opencl").mod.imports[0].inspect_source("cl")

    return generate


def _module(scope):
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((32,), "float16"), B: T.Tensor((16,), "uint32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 1):
                    buf = T.alloc_tensor(ty_args=[T.Tensor((32,), "float16", scope=scope)])
                    for i in range(32):
                        buf[i] = A[i]
                    words = buf.view(T.uint32.dtype)
                    for i in range(16):
                        B[i] = words[i]

    return Module


@pytest.mark.parametrize("scope", ["local", "shared"])
def test_array_declaration_states_its_alignment(opencl_source, scope):
    source = opencl_source(_module(scope))
    declaration = next(line for line in source.splitlines() if "buf_ptr[32]" in line)
    assert "__attribute__((aligned(" in declaration


def test_naturally_aligned_array_is_unchanged(opencl_source):
    """An alignment no larger than the element's own needs no attribute."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((4,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 1):
                    buf = T.alloc_local((4,), "float32", align=4)
                    for i in range(4):
                        buf[i] = A[i]
                    for i in range(4):
                        A[i] = buf[i] + T.float32(1)

    source = opencl_source(Module)
    declaration = next(line for line in source.splitlines() if "buf_ptr[4]" in line)
    assert "aligned" not in declaration


if __name__ == "__main__":
    tvm_tirx.testing.main()
