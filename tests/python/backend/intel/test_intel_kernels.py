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
"""The kernels in ``examples/intel_kernels``, checked against numpy on an Intel GPU."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.intel.target_tags import device_tag
from tvm_tirx.testing import env

KERNELS = Path(__file__).resolve().parents[4] / "examples" / "intel_kernels"
TARGET = device_tag() or "intel/arc-b60-opencl"


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


def _load(name):
    """Import a kernel module by path; TVMScript needs it in sys.modules to read its source."""
    module_name = f"intel_kernels_{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, KERNELS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def test_gemm_plans_the_last_wave():
    gemm = _load("gemm")
    # 2048^3 on the B60's 40 work-groups: 3 whole waves, then 8 tiles in 5 chunks each.
    assert gemm.plan(2048, 2048, 2048, 40) == {
        "tiles": 128,
        "whole": 120,
        "leftover": 8,
        "chunks": 5,
    }
    # Whole waves need no split.
    assert gemm.plan(2560, 2048, 2048, 40)["leftover"] == 0


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("split", [False, True], ids=["whole", "split"])
@pytest.mark.parametrize("label", ["2048^3", "edges 1000^3", "edges 777x520x264"])
def test_gemm(label, split):
    gemm = _load("gemm")
    cfg = next(c for c in gemm.CONFIGS if c["label"] == label)
    data = gemm.prepare_data(**cfg, split=split)
    fun = tvm_tirx.compile(gemm.get_kernel(**cfg, split=split), target=TARGET)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("A", "B", "C", "W")]
        fun["main"](*tensors)
        gemm.check_correctness(dict(data, C=tensors[2].numpy()), **cfg)
        assert np.isfinite(tensors[2].numpy()).all()

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("keys", [16, 32])
@pytest.mark.parametrize("label", ["h8 l256 d64", "h8 l256 d64 causal"])
def test_attention(label, keys):
    attention = _load("attention")
    cfg = next(c for c in attention.CONFIGS if c["label"] == label)
    data = attention.prepare_data(**cfg)
    fun = tvm_tirx.compile(attention.get_kernel(**cfg, keys=keys), target=TARGET)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("Q", "Kw", "V", "O")]
        fun["main"](*tensors)
        attention.check_correctness(dict(data, O=tensors[3].numpy()), **cfg)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("single_pass", [False, True], ids=["two-pass", "single-pass"])
def test_rmsnorm(single_pass):
    rmsnorm = _load("rmsnorm")
    cfg = {"rows": 37, "hidden": 4096}  # not a multiple of the rows per work-group
    data = rmsnorm.prepare_data(**cfg)
    fun = tvm_tirx.compile(rmsnorm.get_kernel(**cfg, single_pass=single_pass), target=TARGET)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("X", "Wt", "Y")]
        fun["main"](*tensors)
        rmsnorm.check_correctness(dict(data, Y=tensors[2].numpy()), **cfg)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
