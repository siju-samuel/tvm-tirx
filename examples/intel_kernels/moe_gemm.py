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
"""MoE grouped GEMM on Intel Xe DPAS: each expert's tokens times that expert's weights.

``Y[t] = X[t] @ W[e].T`` for every token row ``t`` routed to expert ``e``.  The
tokens are sorted by expert, so expert ``e`` owns rows ``offsets[e]`` to
``offsets[e + 1]`` of ``X`` and ``Y``, and the weights are stored ``(E, N, K)``,
the way model weights usually are.  The row counts live on the device, so the
host never waits for them.

Work-groups cover a bound on the number of their row tiles, ``ceil(T / rows) + E``,
across 128-column blocks of N; their tile height grows with the tokens per
expert.  Each one walks the expert offsets to find which
expert's rows its tile index falls in.  Expert boundaries need no masking code:
the kernel addresses ``X`` and ``Y`` through views whose height is the tile's
expert's last row, so the 2D block reads return zeros past it and the block
writes drop rows past it, both in hardware.  A work-group past the last tile
gets a tile at row ``T``, which reads nothing and writes nothing.  The weights
come in through the transposed block read, which hands work-item ``n`` row
``n`` of ``W[e]`` already in the VNNI pairs DPAS reads.  Each step covers 32 of
K and prefetches the next step's tokens and weights, as the dense GEMM does.

The module follows TIRx-kernels' kernel protocol.
"""

from __future__ import annotations

import math

import numpy as np

from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import S, TileLayout

KERNEL_META = {
    "name": "intel_moe_grouped_gemm",
    "category": "moe",
    "runtime_targets": ["intel/arc-b60-opencl", "intel/arc-b60-oneapi"],
}

CONFIGS = [
    {"label": "e8 t1024 n512 k256", "experts": 8, "tokens": 1024, "n": 512, "k": 256},
    {"label": "e64 t2048 n1536 k2048", "experts": 64, "tokens": 2048, "n": 1536, "k": 2048},
    {"label": "e8 t8192 n4096 k2048", "experts": 8, "tokens": 8192, "n": 4096, "k": 2048},
]

NR = 2  # a sub-group's tile is (tile_rows) x 32: two 16-column blocks
SG_N = 4  # sub-groups across N per work-group
TN = NR * 16
WN = SG_N * TN
TK = 32  # K per step: two 16-deep DPAS blocks
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


def get_kernel(
    *, backend_config=None, experts, tokens, n, k, tile_rows=None, sub_groups_m=None, **_
):
    """The grouped GEMM as an IRModule whose ``main(X, Ww, offsets, Y)`` launches one kernel.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    experts, tokens, n, k : int
        Experts, routed token rows, and each expert's ``(n, k)`` weight shape.
        ``n`` must be a multiple of 128 and ``k`` of 32.
    tile_rows : int, optional
        Token rows per sub-group tile, 32 or 64.
    sub_groups_m : int, optional
        Sub-groups stacked along the tokens in a work-group, 1 or 4; four share
        their weight columns through the cache, as the dense GEMM's do.  A
        work-group's tile that runs past its expert's rows wastes the rest, so
        the defaults grow with the average tokens per expert: 32 x 1 below 256,
        64 x 1 below 1024, and 64 x 4 from there.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(X, Ww, offsets, Y)``: ``X`` is ``(tokens, k)`` float16 sorted by
        expert, ``Ww`` the ``(experts, n, k)`` float16 weights as
        ``(experts, n, k // 2)`` uint32 words, ``offsets`` the ``experts + 1``
        int32 row offsets, and ``Y`` the ``(tokens, n)`` float32 output.
    """
    per_expert = tokens / experts
    if tile_rows is None:
        tile_rows = 64 if per_expert >= 256 else 32
    if sub_groups_m is None:
        sub_groups_m = 4 if per_expert >= 1024 and tile_rows == 64 else 1
    if tile_rows not in (32, 64) or sub_groups_m not in (1, 4):
        raise ValueError("tile_rows must be 32 or 64 and sub_groups_m 1 or 4")
    if n % WN or k % TK:
        raise ValueError(f"n must be a multiple of {WN} and k of {TK}")
    E, N, K, TM, SG_M = experts, n, k, tile_rows, sub_groups_m
    MR, RB = TM // 8, TM // 32  # 8-row fragments, 32-row read blocks
    WM = SG_M * TM  # a work-group's rows
    tiles = math.ceil(tokens / WM) + E
    # A: K block outer, then row: register kb * TM + row.  A 32-row read fills four fragments.
    a_read = _lay((RB, 2, 32, 16), (32, TM, 1))
    a_dpas = _lay((MR, 2, 8, 16), (8, TM, 1))
    # B: K block outer, then N block, then word: register kb * NR * 8 + nb * 8 + word.
    b_read = _lay((NR, 2, 8, 16), (8, NR * 8, 1))
    b_dpas = _lay((2, NR, 8, 16), (NR * 8, 8, 1))
    d = _lay((MR, NR, 8, 16), (NR * 8, 8, 1))
    prefetch = "intel_sub_group_2d_block_prefetch_16b_32r16x2c"

    @I.ir_module
    class Module:
        @T.function
        def main(
            X: T.Tensor((tokens, K), "float16"),
            Ww: T.Tensor((E, N, K // 2), "uint32"),
            offsets: T.Tensor((E + 1,), "int32"),
            Y: T.Tensor((tokens, N), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.y", tiles) as g:
                with T.launch_thread("blockIdx.x", N // WN) as nb:
                    with T.launch_thread("threadIdx.x", SG_M * SG_N * 16) as tx:
                        ar = T.alloc_local((2 * TM,), "float16")
                        br = T.alloc_local((2 * NR * 8,), "uint32")
                        dr = T.alloc_local((MR * NR * 8,), "float32")
                        # The tile's first row, its expert's end row, and its expert; a tile
                        # past the last one stays at row T, outside every expert.
                        info = T.alloc_local((4,), "int32")
                        info[0] = tokens
                        info[1] = tokens
                        info[2] = 0
                        info[3] = 0  # tiles before expert e
                        for e in T.serial(E, unroll=False):
                            start = offsets[e]
                            count = (offsets[e + 1] - start + WM - 1) // WM
                            if g >= info[3] and g < info[3] + count:
                                info[0] = start + (g - info[3]) * WM
                                info[1] = offsets[e + 1]
                                info[2] = e
                            info[3] = info[3] + count
                        row0 = info[0] + tx // 16 // SG_N * TM
                        end = info[1]
                        expert = info[2]
                        col0 = nb * WN + tx // 16 % SG_N * TN
                        Xe = T.decl_tensor(
                            X.data, ty_args=[T.Tensor((end, K), "float16", data=X.data)]
                        )
                        Ye = T.decl_tensor(
                            Y.data, ty_args=[T.Tensor((end, N), "float32", data=Y.data)]
                        )
                        # The weights as halves, for the 16-bit prefetch.
                        Wh = T.decl_tensor(
                            Ww.data, ty_args=[T.Tensor((E, N, K), "float16", data=Ww.data)]
                        )
                        for e in range(MR * NR * 8):
                            dr[e] = T.float32(0)
                        for kb in T.serial(K // TK, unroll=False):
                            kk = kb * TK
                            T.intel.tile.block_prefetch(
                                prefetch,
                                Xe[row0 : row0 + TM, kk + TK : kk + 2 * TK],
                                scope="subgroup",
                            )
                            T.intel.tile.block_prefetch(
                                prefetch,
                                Wh[expert, col0 : col0 + TN, kk + TK : kk + 2 * TK],
                                scope="subgroup",
                            )
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_32r16x1c",
                                ar.view(RB, 2, 32, 16, layout=a_read),
                                Xe[row0 : row0 + TM, kk : kk + TK],
                                scope="subgroup",
                            )
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_transpose_32b_16r8x1c",
                                br.view(NR, 2, 8, 16, layout=b_read),
                                Ww[expert, col0 : col0 + TN, kk // 2 : kk // 2 + TK // 2],
                                scope="subgroup",
                            )
                            T.intel.tile.dpas(
                                F16,
                                dr.view(MR, NR, 8, 16, layout=d),
                                ar.view(MR, 2, 8, 16, layout=a_dpas),
                                br.view(2, NR, 8, 16, layout=b_dpas),
                                dr.view(MR, NR, 8, 16, layout=d),
                                scope="subgroup",
                            )
                        T.intel.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            Ye[row0 : row0 + TM, col0 : col0 + TN],
                            dr.view(MR, NR, 8, 16, layout=d),
                            scope="subgroup",
                        )

    return Module


def prepare_data(*, experts, tokens, n, k, seed=0, counts=None, **_):
    """Tokens split among experts unevenly, sorted, with their weights.

    ``counts`` gives each expert's tokens; by default they are drawn unevenly."""
    rng = np.random.default_rng(seed)
    if counts is None:
        share = rng.dirichlet(np.full(experts, 0.5))
        counts = np.floor(share * tokens).astype(np.int64)
        counts[0] += tokens - counts.sum()
    counts = np.asarray(counts, np.int64)
    if len(counts) != experts or counts.sum() != tokens:
        raise ValueError("counts must give each expert's tokens and sum to tokens")
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    w = (rng.standard_normal((experts, n, k)) / math.sqrt(k)).astype(np.float16)
    return {
        "X": rng.standard_normal((tokens, k)).astype(np.float16),
        "W": w,
        "Ww": np.ascontiguousarray(w).view(np.uint32),
        "offsets": offsets,
        "Y": np.zeros((tokens, n), np.float32),
    }


def reference(x, w, offsets):
    """Each expert's rows times its weights, in float32."""
    y = np.zeros((x.shape[0], w.shape[1]), np.float32)
    for e in range(w.shape[0]):
        lo, hi = offsets[e], offsets[e + 1]
        y[lo:hi] = x[lo:hi].astype(np.float32) @ w[e].astype(np.float32).T
    return y


def check_correctness(outputs, **_):
    """Y against a float32 numpy grouped GEMM of the same inputs."""
    want = reference(outputs["X"], outputs["W"], outputs["offsets"])
    np.testing.assert_allclose(outputs["Y"], want, rtol=2e-3, atol=2e-3)


def flops(*, tokens, n, k, **_):
    """Two flops per multiply-add of the routed rows."""
    return 2 * tokens * n * k


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
        tensors = [tvm_tirx.runtime.tensor(data[key], dev) for key in ("X", "Ww", "offsets", "Y")]
        fun["main"](*tensors)
        dev.sync()
        check_correctness(dict(data, Y=tensors[3].numpy()), **cfg)
        line = f"{cfg['label']:26s} correct"
        if args.bench:
            result = bench_interleaved({"tirx": lambda: fun["main"](*tensors)}, dev)
            line += f"  {flops(**cfg) / result['tirx'].median_us / 1e6:.1f} TFLOP/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
