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
"""Loop-unroll annotations reach the generated OpenCL C.

``T.serial(n, unroll=...)`` records what the caller wants, and the OpenCL
compiler needs it said: IGC fully unrolls a constant-trip loop, which on a GEMM
K loop of 32 matrix instructions cost over 40% of its speed on an Arc Pro B60.
"""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _source(monkeypatch, unroll):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((64,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 1):
                    for i in T.serial(64, unroll=unroll):
                        A[i] = A[i] + T.float32(1)

    source = tvm_tirx.compile(Module, target="opencl").mod.imports[0].inspect_source("cl")
    lines = source.splitlines()
    loop = next(n for n, line in enumerate(lines) if line.strip().startswith("for (int i"))
    return lines[loop - 1].strip()


@pytest.mark.parametrize(
    "unroll, pragma",
    [
        (False, "#pragma unroll 1"),
        # A bool must not be read as the count 1, which would disable unrolling.
        (True, "#pragma unroll"),
        (4, "#pragma unroll 4"),
    ],
)
def test_annotation_becomes_a_pragma_before_the_loop(monkeypatch, unroll, pragma):
    assert _source(monkeypatch, unroll) == pragma


def test_unannotated_loop_has_no_pragma(monkeypatch):
    assert not _source(monkeypatch, None).startswith("#pragma")


if __name__ == "__main__":
    tvm_tirx.testing.main()
