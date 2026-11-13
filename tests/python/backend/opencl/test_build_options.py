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
"""OpenCL ``build_options``: compiler options a target asks each kernel to be built with.

Code generation records them as the first line of each kernel's source, the only
per-kernel artifact an OpenCL module carries, and the runtime passes them to
clBuildProgram when it builds that kernel.
"""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.testing import env

# A function-like macro defined only by the build options: the kernel compiles
# and returns 42 if the runtime passed them, and fails to build if it did not.
PROBE = "-Dtirx_build_option_probe()=42"


def _module():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((1,), "int32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 1):
                    A[0] = T.call_pure_extern("tirx_build_option_probe", ty="int32")

    return Module


def _target(options=None):
    return Target({"kind": "opencl", **({"build_options": options} if options else {})})


def test_options_lead_the_kernel_source(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    source = tvm_tirx.compile(_module(), target=_target(PROBE)).mod.imports[0].inspect_source("cl")
    lines = source.splitlines()
    assert lines[0].startswith("// Function: ")
    assert lines[1] == f"// Build options: {PROBE}"


def test_no_options_no_directive(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    source = tvm_tirx.compile(_module(), target=_target()).mod.imports[0].inspect_source("cl")
    assert "// Build options:" not in source


def test_multiline_options_are_rejected(monkeypatch):
    """A newline would end the directive and leave the rest as kernel source."""
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    with pytest.raises(Exception, match="single line"):
        tvm_tirx.compile(_module(), target=_target("-DA=1\n-DB=2"))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_runtime_builds_with_the_options():
    fun = tvm_tirx.compile(_module(), target=_target(PROBE))

    def run_and_check():
        a = tvm_tirx.runtime.tensor(np.zeros(1, np.int32), tvm_tirx.opencl(0))
        fun["main"](a)
        assert a.numpy()[0] == 42

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_kernel_needing_the_options_fails_without_them():
    """Guards the test above: the probe is undefined unless the options were applied."""
    fun = tvm_tirx.compile(_module(), target=_target())

    def run():
        fun["main"](tvm_tirx.runtime.tensor(np.zeros(1, np.int32), tvm_tirx.opencl(0)))

    with pytest.raises(Exception, match="build error"):
        tvm_tirx.testing.run_with_gpu_lock(run)


if __name__ == "__main__":
    tvm_tirx.testing.main()
