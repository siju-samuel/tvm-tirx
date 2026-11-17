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
"""RMSNorm on an Intel Xe GPU: ``y = x / sqrt(mean(x^2) + eps) * w``, float16 in and out.

Rows are read with 1D sub-group block reads, 128 elements in one 256-byte
message where a load per work-item would move 2 bytes each, and written back
with the matching block writes.  Each work-item sums the squares of the
elements it holds, and ``T.intel.tile.reduce`` adds the 16 partial sums across
its sub-group, leaving the total on every work-item.

By default four sub-groups share a row: each keeps its quarter in registers,
and their totals meet in shared local memory behind a work-group barrier, so
the row is read once.  ``single_pass=False``, or a hidden size that is not a
multiple of 512, gives each row one sub-group that reads it twice, the second
time from cache.

The module follows TIRx-kernels' kernel protocol.
"""

from __future__ import annotations

import numpy as np

from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import R, S, TileLayout

KERNEL_META = {
    "name": "intel_rmsnorm",
    "category": "norm",
    "runtime_targets": ["intel/arc-b60-opencl"],
}

CONFIGS = [
    {"label": "rows 64 hidden 4096", "rows": 64, "hidden": 4096},
    {"label": "rows 4096 hidden 4096", "rows": 4096, "hidden": 4096},
    {"label": "rows 8192 hidden 7168", "rows": 8192, "hidden": 7168},
]

EPS = 1e-6
SUB_GROUPS = 8  # rows per work-group
PER_LANE = 8  # elements each work-item reads per step

# 128 contiguous elements: element 16 c + i in component c of work-item i, one 1D block read.
CHUNK = TileLayout(S[(1, 1, PER_LANE, 16) : (PER_LANE, PER_LANE, 1, 1 @ sglid)])
READ = f"intel_sub_group_block_read_us{PER_LANE}"
WRITE = f"intel_sub_group_block_write_us{PER_LANE}"
# Each work-item's partial sum, and the row's total on every work-item.
PARTIAL = TileLayout(S[(1, 1, 1, 16) : (1, 1, 1, 1 @ sglid)])
TOTAL = TileLayout(S[(1, 1, 1, 1) : (1, 1, 1, 1)] + R[16 : 1 @ sglid])


@T.function(private=True)
def _inverse_rms(total: T.float32, inv_hidden: T.float32) -> T.float32:
    return T.rsqrt(total * inv_hidden + T.float32(EPS))


SPLIT = 4  # sub-groups per row in the single-pass form


def _single_pass(rows, hidden):
    """A work-group of SPLIT sub-groups per row, each keeping its slice in registers."""
    piece = hidden // SPLIT
    runs = piece // (16 * PER_LANE)
    slice_layout = TileLayout(
        S[(1, runs, PER_LANE, 16) : (runs * PER_LANE, PER_LANE, 1, 1 @ sglid)]
    )

    @I.ir_module
    class Module:
        @T.function
        def main(
            X: T.Tensor((rows, hidden), "float16"),
            Wt: T.Tensor((hidden,), "float16"),
            Y: T.Tensor((rows, hidden), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", rows) as row:
                with T.launch_thread("threadIdx.x", SPLIT * 16) as tx:
                    x = T.alloc_local((runs * PER_LANE,), "float16")
                    w = T.alloc_local((runs * PER_LANE,), "float16")
                    partial = T.alloc_local((1,), "float32")
                    total = T.alloc_local((1,), "float32")
                    scale = T.alloc_local((1,), "float32")
                    sums = T.alloc_shared((SPLIT,), "float32")
                    sg = tx // 16
                    T.intel.tile.block_read(
                        READ,
                        x.view(1, runs, PER_LANE, 16, layout=slice_layout),
                        X[row, sg * piece : sg * piece + piece],
                        scope="subgroup",
                    )
                    partial[0] = T.float32(0)
                    for e in T.unroll(runs * PER_LANE):
                        v = T.Cast("float32", x[e])
                        partial[0] = partial[0] + v * v
                    T.intel.tile.reduce(
                        "sum",
                        total.view(1, 1, 1, 1, layout=TOTAL),
                        partial.view(1, 1, 1, 16, layout=PARTIAL),
                    )
                    # Every work-item of a sub-group holds the same total, so all may store it.
                    sums[sg] = total[0]
                    T.gpu_storage_sync("shared")
                    total[0] = T.float32(0)
                    for j in T.unroll(SPLIT):
                        total[0] = total[0] + sums[j]
                    T.intel.tile.ewise(
                        _inverse_rms,
                        scale.view(1, 1, 1, 1, layout=TOTAL),
                        total.view(1, 1, 1, 1, layout=TOTAL),
                        T.float32(1.0 / hidden),
                    )
                    T.intel.tile.block_read(
                        READ,
                        w.view(1, runs, PER_LANE, 16, layout=slice_layout),
                        Wt[sg * piece : sg * piece + piece],
                        scope="subgroup",
                    )
                    for e in T.unroll(runs * PER_LANE):
                        x[e] = T.Cast(
                            "float16", T.Cast("float32", x[e]) * scale[0] * T.Cast("float32", w[e])
                        )
                    T.intel.tile.block_write(
                        WRITE,
                        Y[row, sg * piece : sg * piece + piece],
                        x.view(1, runs, PER_LANE, 16, layout=slice_layout),
                        scope="subgroup",
                    )

    return Module


def get_kernel(*, backend_config=None, rows, hidden, single_pass=True, **_):
    """RMSNorm as an IRModule whose ``main(X, Wt, Y)`` launches one kernel.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    rows, hidden : int
        The input's shape.  ``hidden`` must be a multiple of 128.
    single_pass : bool
        With a ``hidden`` that is a multiple of 512, read each row once: four
        sub-groups share it, each keeps its slice in registers, and their sums
        meet in shared local memory.  Otherwise one sub-group reads its row twice.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(X, Wt, Y)``: X and Y are ``(rows, hidden)`` and Wt is ``(hidden,)``,
        all float16.
    """
    step = 16 * PER_LANE
    if hidden % step != 0:
        raise ValueError(f"hidden must be a multiple of {step}")
    if single_pass and hidden % (SPLIT * step) == 0:
        return _single_pass(rows, hidden)
    groups = (rows + SUB_GROUPS - 1) // SUB_GROUPS

    @I.ir_module
    class Module:
        @T.function
        def main(
            X: T.Tensor((rows, hidden), "float16"),
            Wt: T.Tensor((hidden,), "float16"),
            Y: T.Tensor((rows, hidden), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", groups) as b:
                with T.launch_thread("threadIdx.x", SUB_GROUPS * 16) as tx:
                    x = T.alloc_local((PER_LANE,), "float16")
                    w = T.alloc_local((PER_LANE,), "float16")
                    y = T.alloc_local((PER_LANE,), "float16")
                    partial = T.alloc_local((1,), "float32")
                    total = T.alloc_local((1,), "float32")
                    scale = T.alloc_local((1,), "float32")
                    row = T.min(b * SUB_GROUPS + tx // 16, rows - 1)
                    partial[0] = T.float32(0)
                    for c in T.serial(hidden // step, unroll=False):
                        T.intel.tile.block_read(
                            READ,
                            x.view(1, 1, PER_LANE, 16, layout=CHUNK),
                            X[row, c * step : c * step + step],
                            scope="subgroup",
                        )
                        for i in T.unroll(PER_LANE):
                            v = T.Cast("float32", x[i])
                            partial[0] = partial[0] + v * v
                    T.intel.tile.reduce(
                        "sum",
                        total.view(1, 1, 1, 1, layout=TOTAL),
                        partial.view(1, 1, 1, 16, layout=PARTIAL),
                    )
                    T.intel.tile.ewise(
                        _inverse_rms,
                        scale.view(1, 1, 1, 1, layout=TOTAL),
                        total.view(1, 1, 1, 1, layout=TOTAL),
                        T.float32(1.0 / hidden),
                    )
                    # The last work-group's spare sub-groups repeat the last row; only one writes.
                    if b * SUB_GROUPS + tx // 16 < rows:
                        for c in T.serial(hidden // step, unroll=False):
                            T.intel.tile.block_read(
                                READ,
                                x.view(1, 1, PER_LANE, 16, layout=CHUNK),
                                X[row, c * step : c * step + step],
                                scope="subgroup",
                            )
                            T.intel.tile.block_read(
                                READ,
                                w.view(1, 1, PER_LANE, 16, layout=CHUNK),
                                Wt[c * step : c * step + step],
                                scope="subgroup",
                            )
                            for i in T.unroll(PER_LANE):
                                y[i] = T.Cast(
                                    "float16",
                                    T.Cast("float32", x[i]) * scale[0] * T.Cast("float32", w[i]),
                                )
                            T.intel.tile.block_write(
                                WRITE,
                                Y[row, c * step : c * step + step],
                                y.view(1, 1, PER_LANE, 16, layout=CHUNK),
                                scope="subgroup",
                            )

    return Module


def prepare_data(*, rows, hidden, **_):
    """Random X and weights, and a zeroed output."""
    rng = np.random.default_rng(0)
    return {
        "X": rng.standard_normal((rows, hidden)).astype(np.float16),
        "Wt": (1 + 0.1 * rng.standard_normal(hidden)).astype(np.float16),
        "Y": np.zeros((rows, hidden), np.float16),
    }


def check_correctness(outputs, **_):
    """Y against a float32 numpy RMSNorm of the same inputs."""
    x = outputs["X"].astype(np.float32)
    want = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * outputs["Wt"]
    np.testing.assert_allclose(outputs["Y"].astype(np.float32), want, rtol=1e-2, atol=1e-2)


def bytes_moved(*, rows, hidden, **_):
    """The bytes the kernel must read and write once: X, Y and the weights."""
    return 2 * (2 * rows * hidden + hidden)


def main(argv=None):
    """Check every config, or the named ones, and with ``--bench`` time them."""
    import argparse  # pylint: disable=import-outside-toplevel

    import tvm_tirx  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.intel.target_tags import (
        device_tag,  # pylint: disable=import-outside-toplevel
    )
    from tvm_tirx.testing.device_bench import (
        bench_interleaved,  # pylint: disable=import-outside-toplevel
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="*", help="config labels; all by default")
    parser.add_argument("--bench", action="store_true", help="time each config")
    parser.add_argument("--device", type=int, default=0, help="OpenCL device index")
    args = parser.parse_args(argv)
    dev = tvm_tirx.opencl(args.device)
    target = device_tag(args.device) or KERNEL_META["runtime_targets"][0]
    for cfg in CONFIGS:
        if args.labels and cfg["label"] not in args.labels:
            continue
        data = prepare_data(**cfg)
        fun = tvm_tirx.compile(get_kernel(**cfg), target=target)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("X", "Wt", "Y")]
        fun["main"](*tensors)
        dev.sync()
        check_correctness(dict(data, Y=tensors[2].numpy()), **cfg)
        line = f"{cfg['label']:24s} correct"
        if args.bench:
            result = bench_interleaved({"tirx": lambda: fun["main"](*tensors)}, dev)
            line += f"  {bytes_moved(**cfg) / result['tirx'].median_us / 1e3:.0f} GB/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
