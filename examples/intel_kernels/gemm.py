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
"""fp16 GEMM on Intel Xe DPAS, with the last wave of tiles split along K.

``C[M, N] = A[M, K] @ B[K, N]`` in float32.  Each sub-group computes a 64 x 32
tile of C with DPAS from 2D block reads, prefetching one K step ahead; a
work-group of 4 x 4 sub-groups computes a 256 x 128 tile.  A size that is not a
multiple of the tile reads zeros past the edge and has its stores clipped, so
M is free, and N and K need only keep each row's pitch a multiple of 16 bytes.

A launch of more tiles than the GPU runs at once goes in waves, and a last
wave that is only partly full leaves cores idle: on an Arc Pro B60, which runs
40 of these work-groups, 2048^3's 128 tiles take 3.2 waves.  So the tiles that
fill whole waves run one per work-group, and the leftover tiles are each split
into K chunks, enough to fill one more wave.  Each chunk writes its partial
tile to a workspace, and a second kernel adds the chunks of each tile into C.
The kernel boundary orders the two, so no work-group waits on another.

The module follows TIRx-kernels' kernel protocol: KERNEL_META, CONFIGS,
get_kernel, prepare_data and check_correctness.
"""

from __future__ import annotations

import math

import numpy as np

from tvm_tirx.backend.intel.occupancy import ARC_B60, resident_work_groups
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import S, TileLayout

KERNEL_META = {
    "name": "intel_fp16_gemm",
    "category": "gemm",
    "runtime_targets": ["intel/arc-b60-opencl"],
}

CONFIGS = [
    {"label": "2048^3", "M": 2048, "N": 2048, "K": 2048},
    {"label": "4096^3", "M": 4096, "N": 4096, "K": 4096},
    {"label": "2560x2048x2048", "M": 2560, "N": 2048, "K": 2048},
    {"label": "edges 1000^3", "M": 1000, "N": 1000, "K": 1000},
    {"label": "edges 777x520x264", "M": 777, "N": 520, "K": 264},
]

# The sub-group and work-group tiles, and the K step.
TM, TN, TK = 64, 32, 32
SG_M, SG_N = 4, 4
WM, WN = SG_M * TM, SG_N * TN
SUB_GROUPS = SG_M * SG_N


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


A_READ = _lay((2, 2, 32, 16), (32, 64, 1))  # two 32-row blocks of A for each 16-wide K half
A_DPAS = _lay((8, 2, 8, 16), (8, 64, 1))  # the same registers as eight 8-row fragments
B_READ = _lay((1, 2, 16, 16), (32, 16, 1))
B_DPAS = _lay((2, 2, 8, 16), (8, 16, 1))
D = _lay((8, 2, 8, 16), (16, 8, 1))


def plan(M, N, K, resident):
    """How the tiles are scheduled: whole-wave tiles, leftover tiles, and K chunks per leftover.

    Parameters
    ----------
    M, N, K : int
        The problem.
    resident : int
        Work-groups the GPU runs at once.

    Returns
    -------
    plan : dict
        ``tiles``, ``whole`` (tiles run one per work-group), ``leftover`` and ``chunks``.
    """
    tiles = math.ceil(M / WM) * math.ceil(N / WN)
    steps = math.ceil(K / TK)
    leftover = tiles % resident
    whole = tiles - leftover
    # Enough chunks for the leftover tiles to fill one wave, each at least one K step.
    chunks = max(1, min(resident // leftover, steps)) if leftover else 1
    if chunks == 1:
        whole, leftover = tiles, 0
    return {"tiles": tiles, "whole": whole, "leftover": leftover, "chunks": chunks}


def get_kernel(*, backend_config=None, M, N, K, resident=None, split=True, **_):
    """The GEMM as an IRModule whose ``main(A, B, C, W)`` launches one or two kernels.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    M, N, K : int
        The problem.
    resident : int, optional
        Work-groups the GPU runs at once; the B60's 40 by default.
    split : bool
        Split the leftover tiles along K.  ``False`` runs every tile whole.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(A, B, C, W)``; ``W`` is the workspace, of :func:`workspace_shape`.
    """
    if resident is None:
        resident = resident_work_groups(ARC_B60, sub_groups=SUB_GROUPS, registers=256)
    if split:
        p = plan(M, N, K, resident)
        whole, leftover, chunks = p["whole"], p["leftover"], p["chunks"]
    else:
        whole, leftover, chunks = math.ceil(M / WM) * math.ceil(N / WN), 0, 1
    NB = math.ceil(N / WN)
    steps = math.ceil(K / TK)
    groups = whole + leftover * chunks
    slots = max(1, leftover * chunks)
    fix_threads = 256
    fix_groups = max(1, math.ceil(leftover * WM * WN / fix_threads))

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            C: T.Tensor((M, N), "float32"),
            W: T.Tensor((slots * WM, WN), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", groups) as wg:
                with T.launch_thread("threadIdx.x", SUB_GROUPS * 16) as tx:
                    ar = T.alloc_local((128,), "float16")
                    br = T.alloc_local((64,), "float16")
                    dr = T.alloc_local((128,), "float32")
                    piece = wg - whole
                    tile = T.if_then_else(wg < whole, wg, whole + piece // chunks)
                    chunk = T.if_then_else(wg < whole, 0, piece % chunks)
                    k_begin = chunk * steps // chunks
                    k_end = T.if_then_else(wg < whole, steps, (chunk + 1) * steps // chunks)
                    sg_row = tx // 16 // SG_N * TM
                    sg_col = tx // 16 % SG_N * TN
                    row0 = tile // NB * WM + sg_row
                    col0 = tile % NB * WN + sg_col
                    for e in range(128):
                        dr[e] = T.float32(0)
                    for kb in T.serial(k_end - k_begin, unroll=False):
                        k = (k_begin + kb) * TK
                        T.intel.tile.block_prefetch(
                            "intel_sub_group_2d_block_prefetch_16b_32r16x2c",
                            A[row0 : row0 + TM, k + TK : k + 2 * TK],
                            scope="subgroup",
                        )
                        T.intel.tile.block_prefetch(
                            "intel_sub_group_2d_block_prefetch_16b_32r16x2c",
                            B[k + TK : k + 2 * TK, col0 : col0 + TN],
                            scope="subgroup",
                        )
                        T.intel.tile.block_read(
                            "intel_sub_group_2d_block_read_16b_32r16x1c",
                            ar.view(2, 2, 32, 16, layout=A_READ),
                            A[row0 : row0 + TM, k : k + TK],
                            scope="subgroup",
                        )
                        T.intel.tile.block_read(
                            "intel_sub_group_2d_block_read_transform_16b_32r16x1c",
                            br.view(T.uint32.dtype).view(1, 2, 16, 16, layout=B_READ),
                            B[k : k + TK, col0 : col0 + TN],
                            scope="subgroup",
                        )
                        T.intel.tile.dpas(
                            "intel_sub_group_f16_f16_matrix_mad_k16",
                            dr.view(8, 2, 8, 16, layout=D),
                            ar.view(8, 2, 8, 16, layout=A_DPAS),
                            br.view(T.uint32.dtype).view(2, 2, 8, 16, layout=B_DPAS),
                            dr.view(8, 2, 8, 16, layout=D),
                            scope="subgroup",
                        )
                    if wg < whole:
                        T.intel.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            C[row0 : row0 + TM, col0 : col0 + TN],
                            dr.view(8, 2, 8, 16, layout=D),
                            scope="subgroup",
                        )
                    else:
                        T.intel.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            W[piece * WM + sg_row : piece * WM + sg_row + TM, sg_col : sg_col + TN],
                            dr.view(8, 2, 8, 16, layout=D),
                            scope="subgroup",
                        )
            if leftover > 0:
                # Add each leftover tile's chunks into C.
                with T.launch_thread("blockIdx.x", fix_groups) as fb:
                    with T.launch_thread("threadIdx.x", fix_threads) as ft:
                        e = fb * fix_threads + ft
                        t = e // (WM * WN)
                        r = e % (WM * WN) // WN
                        c = e % WN
                        row = (whole + t) // NB * WM + r
                        col = (whole + t) % NB * WN + c
                        if t < leftover and row < M and col < N:
                            acc = T.alloc_local((1,), "float32")
                            acc[0] = T.float32(0)
                            for j in range(chunks):
                                acc[0] = acc[0] + W[(t * chunks + j) * WM + r, c]
                            C[row, col] = acc[0]

    return Module


def workspace_shape(M, N, K, resident=None, split=True):
    """The shape of ``main``'s workspace ``W`` for a problem."""
    if resident is None:
        resident = resident_work_groups(ARC_B60, sub_groups=SUB_GROUPS, registers=256)
    p = plan(M, N, K, resident) if split else {"leftover": 0, "chunks": 1}
    return (max(1, p["leftover"] * p["chunks"]) * WM, WN)


def prepare_data(*, M, N, K, resident=None, split=True, **_):
    """Random inputs, a zeroed output and the workspace, as numpy arrays."""
    rng = np.random.default_rng(0)
    return {
        "A": rng.standard_normal((M, K)).astype(np.float16),
        "B": rng.standard_normal((K, N)).astype(np.float16),
        "C": np.zeros((M, N), np.float32),
        "W": np.zeros(workspace_shape(M, N, K, resident, split), np.float32),
    }


def check_correctness(outputs, *, M, N, K, **_):
    """C against a float32 numpy product of the same inputs."""
    a, b = outputs["A"].astype(np.float32), outputs["B"].astype(np.float32)
    want = a @ b
    got = outputs["C"]
    err = float(np.max(np.abs(got - want)) / max(1e-6, float(np.max(np.abs(want)))))
    assert err < 1e-3, f"relative error {err:.2e}"


def main(argv=None):
    """Check every config, or the named ones, and with ``--bench`` time them interleaved."""
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
    parser.add_argument("--bench", action="store_true", help="time whole vs split tiles")
    parser.add_argument("--device", type=int, default=0, help="OpenCL device index")
    args = parser.parse_args(argv)
    dev = tvm_tirx.opencl(args.device)
    target = device_tag(args.device) or KERNEL_META["runtime_targets"][0]
    resident = resident_work_groups(ARC_B60, sub_groups=SUB_GROUPS, registers=256)
    for cfg in CONFIGS:
        if args.labels and cfg["label"] not in args.labels:
            continue
        launches = {}
        for split in (False, True):
            data = prepare_data(**cfg, split=split)
            fun = tvm_tirx.compile(get_kernel(**cfg, split=split), target=target)
            tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("A", "B", "C", "W")]
            fun["main"](*tensors)
            dev.sync()
            check_correctness(dict(data, C=tensors[2].numpy()), **cfg)
            launches["split" if split else "whole"] = lambda f=fun, t=tensors: f["main"](*t)
        p = plan(cfg["M"], cfg["N"], cfg["K"], resident)
        line = (
            f"{cfg['label']:20s} {p['tiles']} tiles, {p['leftover']} leftover"
            f" in {p['chunks']} chunks: correct"
        )
        if args.bench:
            flops = 2 * cfg["M"] * cfg["N"] * cfg["K"]
            results = bench_interleaved(launches, dev)
            line += "".join(
                f"  {name} {flops / m.median_us / 1e6:.1f} TFLOP/s" for name, m in results.items()
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
