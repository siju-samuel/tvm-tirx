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
"""kDLOneAPI devices: OpenCL C run through SYCL, on PyTorch's XPU tensors and streams."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.oneapi import ONEAPI_DEVICE_TYPE, runtime_enabled
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

TARGET = "intel/arc-b60-oneapi"
N = 1 << 16


def _torch_xpu():
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return torch if hasattr(torch, "xpu") and torch.xpu.is_available() else None


def requires_oneapi(test):
    """A GPU test that needs the oneAPI runtime and a PyTorch that sees XPU devices."""
    skip = pytest.mark.skipif(
        not runtime_enabled() or _torch_xpu() is None,
        reason="needs the oneAPI runtime (TVM_TIRX_USE_ONEAPI=ON) and PyTorch with XPU",
    )
    return pytest.mark.gpu(skip(test))


def _add():
    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", N // 256) as bx:
                with T.launch_thread("threadIdx.x", 256) as tx:
                    C[bx * 256 + tx] = A[bx * 256 + tx] + B[bx * 256 + tx] * T.float32(2)

    return Module


def test_oneapi_tag_is_an_opencl_target_on_oneapi_devices():
    target = Target(TARGET)
    assert target.kind.name == "opencl"
    assert int(target.attrs["target_device_type"]) == ONEAPI_DEVICE_TYPE
    assert {"intel", "oneapi"} <= {str(k) for k in target.keys}
    # The same part through OpenCL: same limits and compiler options.
    opencl = Target("intel/arc-b60-opencl")
    for key in ("max_shared_memory_per_block", "thread_warp_size", "build_options"):
        assert target.attrs[key] == opencl.attrs[key]


def test_kernels_compile_to_a_oneapi_module(monkeypatch):
    """The OpenCL C is the same; only the module that runs it changes."""
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    oneapi = tvm_tirx.compile(_add(), target=TARGET).mod.imports[0]
    opencl = tvm_tirx.compile(_add(), target="intel/arc-b60-opencl").mod.imports[0]
    assert oneapi.kind == "oneapi"
    assert opencl.kind == "opencl"
    assert oneapi.inspect_source("cl") == opencl.inspect_source("cl")


@requires_oneapi
def test_device_reports_its_limits():
    dev = tvm_tirx.oneapi(0)
    assert dev.exist
    assert "Intel" in dev.device_name
    assert dev.max_shared_memory_per_block > 0
    assert Target.from_device(dev).attrs["thread_warp_size"] == 16


@requires_oneapi
def test_tirx_tensors_round_trip():
    dev = tvm_tirx.oneapi(0)
    data = np.random.default_rng(0).standard_normal((3, 5)).astype("float32")
    np.testing.assert_array_equal(tvm_tirx.runtime.tensor(data, dev).numpy(), data)


@requires_oneapi
def test_kernel_runs_on_torch_tensors_in_place():
    torch = _torch_xpu()
    fun = tvm_tirx.compile(_add(), target=TARGET)
    a, b = torch.randn(N, device="xpu"), torch.randn(N, device="xpu")
    c = torch.zeros(N, device="xpu")
    pointer = c.data_ptr()
    fun["main"](a, b, c)
    torch.xpu.synchronize()
    assert c.data_ptr() == pointer
    torch.testing.assert_close(c, a + 2 * b)


@requires_oneapi
def test_kernel_runs_in_order_on_a_torch_stream():
    """Under use_torch_stream, a TIRx kernel between two torch ops on a side stream sees the
    first op's result, and the second sees the kernel's, with no synchronization between."""
    import tvm_ffi  # pylint: disable=import-outside-toplevel

    torch = _torch_xpu()
    fun = tvm_tirx.compile(_add(), target=TARGET)
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream), tvm_ffi.use_torch_stream(torch.xpu.stream(stream)):
        results = []
        for trial in range(5):
            a = torch.full((N,), float(trial), device="xpu")
            a.mul_(3.0)
            b = torch.ones(N, device="xpu")
            c = torch.empty(N, device="xpu")
            fun["main"](a, b, c)
            results.append((c + 1.0, 3.0 * trial + 3.0))
    stream.synchronize()
    for got, want in results:
        torch.testing.assert_close(got, torch.full_like(got, want))


def _gemm_module():
    path = Path(__file__).resolve().parents[4] / "examples" / "intel_kernels" / "gemm.py"
    if "intel_kernels_gemm" not in sys.modules:
        spec = importlib.util.spec_from_file_location("intel_kernels_gemm", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["intel_kernels_gemm"] = module
        spec.loader.exec_module(module)
    return sys.modules["intel_kernels_gemm"]


@requires_oneapi
@pytest.mark.parametrize("shape", [(512, 512, 512), (777, 520, 264)])
def test_dpas_gemm_on_torch_tensors(shape):
    """The Intel tile instructions run unchanged: DPAS, 2D block I/O and the split K tail."""
    torch = _torch_xpu()
    gemm = _gemm_module()
    m, n, k = shape
    cfg = {"M": m, "N": n, "K": k}
    fun = tvm_tirx.compile(gemm.get_kernel(**cfg), target=TARGET)
    devices = range(min(2, torch.xpu.device_count()))
    for index in devices:
        a = torch.randn(m, k, dtype=torch.float16, device=f"xpu:{index}")
        b = torch.randn(k, n, dtype=torch.float16, device=f"xpu:{index}")
        c = torch.zeros(m, n, dtype=torch.float32, device=f"xpu:{index}")
        w = torch.zeros(gemm.workspace_shape(**cfg), dtype=torch.float32, device=f"xpu:{index}")
        fun["main"](a, b, c, w)
        torch.xpu.synchronize(index)
        torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm_tirx.testing.main()
