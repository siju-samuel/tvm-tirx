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


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("tile", [(32, 1), (64, 1), (64, 4)], ids=["32x1", "64x1", "64x4"])
def test_moe_grouped_gemm(tile):
    """Uneven experts, some with no tokens: rows stay within their expert in hardware."""
    moe = _load("moe_gemm")
    cfg = {"experts": 8, "tokens": 700, "n": 256, "k": 128}
    data = moe.prepare_data(**cfg, counts=[0, 300, 0, 1, 95, 0, 200, 104])
    rows, sub_groups = tile
    fun = tvm_tirx.compile(
        moe.get_kernel(**cfg, tile_rows=rows, sub_groups_m=sub_groups), target=TARGET
    )

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("X", "Ww", "offsets", "Y")]
        fun["main"](*tensors)
        moe.check_correctness(dict(data, Y=tensors[3].numpy()), **cfg)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_w4a16_packing_round_trips():
    """The tiled layout holds the same words, and dequantizing undoes quantizing."""
    w4 = _load("w4a16_gemm")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 256)).astype(np.float32)
    packed, scale = w4.quantize(w)
    tiled, scales = w4.pack(packed, scale)
    assert tiled.shape == (2, 4, 128) and scales.shape == (2, 32)
    # Word j of column i in step s of column block b is words[16 b + i, 8 s + j].
    assert tiled[1, 2, 16 * 5 + 3] == packed[16 * 1 + 3, 8 * 2 + 5]
    back = w4.dequantize(packed, scale)
    assert np.abs(back - w).max() <= np.repeat(scale, w4.GROUP, axis=1).max() / 2 + 1e-6


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("m", [1, 13, 32])
def test_w4a16_gemm(m, splits):
    w4 = _load("w4a16_gemm")
    cfg = {"m": m, "n": 256, "k": 512}
    data = w4.prepare_data(**cfg, splits=splits)
    fun = tvm_tirx.compile(w4.get_kernel(**cfg, splits=splits), target=TARGET)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("X", "Wt", "Sc", "Y", "P")]
        fun["main"](*tensors)
        w4.check_correctness(dict(data, Y=tensors[3].numpy()), **cfg)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_decode_attention_plans_its_shares():
    dec = _load("decode_attention")

    def plan(batch, max_len, head_dim=128):
        return dec.plan(batch=batch, kv_heads=8, head_dim=head_dim, max_len=max_len)

    # Short shares stay in one work-group and merge in shared local memory.
    assert plan(4, 300) == (1, 8)
    # Long ones go to separate work-groups, 256 shares in all.
    assert plan(8, 8192) == (4, 1)
    assert plan(1, 32768) == (32, 1)
    # Enough sequences and heads need no sharing.
    assert plan(32, 2048) == (1, 1)
    # A work-group's shares fit in 64 KiB of shared local memory: 4 of head_dim 512,
    # so 9 short shares take 3 work-groups.
    assert plan(2, 300, head_dim=512) == (3, 3)


@pytest.mark.gpu
@requires_intel_gpu
@pytest.mark.parametrize("plan", [(1, 1), (3, 1), (1, 4), (2, 3)], ids=lambda p: f"{p[0]}x{p[1]}")
@pytest.mark.parametrize(
    "heads, kv_heads, head_dim", [(32, 8, 128), (28, 4, 64)], ids=["gqa4 d128", "gqa7 d64"]
)
def test_decode_attention(plan, heads, kv_heads, head_dim):
    """Every way of sharing the pages, including empty shares and rows past the group."""
    dec = _load("decode_attention")
    splits, sub_groups = plan
    cfg = {
        "batch": 4,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_len": 200,
        "splits": splits,
        "sub_groups": sub_groups,
    }
    # A full sequence, one token, one whole page and a ragged one.
    data = dec.prepare_data(**cfg, lengths=[200, 1, 16, 77])
    fun = tvm_tirx.compile(dec.get_kernel(**cfg), target=TARGET)

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in dec.ARGS]
        fun["main"](*tensors)
        dec.check_correctness(dict(data, O=tensors[5].numpy()), **cfg)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
