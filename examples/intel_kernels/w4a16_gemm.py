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
"""Weight-only int4 GEMM on Intel Xe DPAS: ``Y = X @ dequant(W).T`` with float16 X.

Decoding a token reads every weight once, so its matrix products are bound by
the bytes of weights; int4 weights move a quarter of float16's.  The weights are
symmetric signed int4 for an ``(N, K)`` matrix, as model weights are stored, with
one float16 scale per group of 128 along K.  :func:`pack` lays them out for the
kernel, once, as int4 kernels usually do:

* A value ``q`` is the nibble ``q + 8``, eight to a ``uint32`` word along K, and
  element ``8 w + i`` of a row sits in word ``w`` at bit ``4 (i // 2) +
  16 (i % 2)``.  Each pair of neighbours is then one mask away from two float16
  values: OR-ing them under 0x6400 makes ``1024 + q + 8``, and a half2 subtract
  and multiply give the pair scaled, already as the VNNI word DPAS reads.
* Each 16 columns by 64 of K, the weights one sub-group needs per step, are 512
  contiguous bytes, word-major, so one 1D sub-group block read hands work-item
  ``n`` the eight words of column ``n``.
* The scales are ``(K / 128, N)``, so the sub-group's 16 scales are contiguous.

Each sub-group computes 16 output columns.  The few rows of X come in through 2D
block reads, which return zeros past the last row, so up to 32 rows run in one
pass.  A decode product has too few output tiles to fill the GPU, so K is split
among work-groups, and a second kernel adds the partial sums.

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
    "name": "intel_w4a16_gemm",
    "category": "gemm",
    "runtime_targets": ["intel/arc-b60-opencl", "intel/arc-b60-oneapi"],
}

CONFIGS = [
    {"label": "m1 n4096 k4096", "m": 1, "n": 4096, "k": 4096},
    {"label": "m8 n4096 k4096", "m": 8, "n": 4096, "k": 4096},
    {"label": "m32 n4096 k4096", "m": 32, "n": 4096, "k": 4096},
    {"label": "m8 n11008 k4096", "m": 8, "n": 11008, "k": 4096},
]

GROUP = 128  # K per scale
STEP = 64  # K per step: one transposed read gives each work-item 8 words, 64 int4 values
SUB_GROUPS = 8  # sub-groups per work-group, 16 columns each
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


def get_kernel(*, backend_config=None, m, n, k, splits=None, **_):
    """The GEMM as an IRModule whose ``main(X, Wt, Sc, Y, P)`` launches one or two kernels.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    m, n, k : int
        ``X`` is ``(m, k)`` with ``m`` at most 32, and the weights ``(n, k)``;
        ``n`` must be a multiple of 128 and ``k`` of 128.
    splits : int, optional
        Work-groups sharing each output tile's K, whose partial sums a second
        kernel adds.  See :func:`_splits` for the default.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(X, Wt, Sc, Y, P)``: ``X`` is ``(m, k)`` float16, ``Wt`` the
        ``(n // 16, k // 64, 128)`` uint32 weights of :func:`pack`, ``Sc`` the
        ``(k // 128, n)`` float16 scales, ``Y`` the ``(m, n)`` float32 output, and
        ``P`` the float32 partial sums, of :func:`partials_shape`.
    """
    if m > 32 or n % (SUB_GROUPS * 16) or k % GROUP:
        raise ValueError("m must be at most 32, n a multiple of 128 and k of 128")
    splits = _splits(m, n, k, splits)
    groups_n = n // (SUB_GROUPS * 16)
    MR = math.ceil(m / 8)  # 8-row fragments of X
    TM = MR * 8
    steps = k // STEP // splits
    a_read = _lay((MR, 4, 8, 16), (32, 8, 1))  # register frag * 32 + kb * 8 + row
    a_dpas = _lay((MR, 4, 8, 16), (32, 8, 1))  # two K blocks' rows adjacent, for x2c reads
    w_read = _lay((1, 1, 8, 16), (8, 8, 1))  # the 8 packed words of each work-item's column
    b_dpas = _lay((4, 1, 8, 16), (8, 8, 1))  # VNNI words: register kb * 8 + word
    d = _lay((MR, 1, 8, 16), (8, 8, 1))
    total = splits * TM
    reduce_threads = 256

    @I.ir_module
    class Module:
        @T.function
        def main(
            X: T.Tensor((m, k), "float16"),
            Wt: T.Tensor((n // 16, k // STEP, 128), "uint32"),
            Sc: T.Tensor((k // GROUP, n), "float16"),
            Y: T.Tensor((m, n), "float32"),
            P: T.Tensor((total, n), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.y", splits) as sp:
                with T.launch_thread("blockIdx.x", groups_n) as gx:
                    with T.launch_thread("threadIdx.x", SUB_GROUPS * 16) as tx:
                        ar = T.alloc_local((MR * 4 * 8,), "float16")
                        wr = T.alloc_local((8,), "uint32")
                        br = T.alloc_local((4 * 8,), "uint32")
                        dr = T.alloc_local((MR * 8,), "float32")
                        scale = T.alloc_local((1,), "float16")
                        lane = tx % 16
                        col0 = (gx * SUB_GROUPS + tx // 16) * 16
                        for e in range(MR * 8):
                            dr[e] = T.float32(0)
                        for s in T.serial(steps, unroll=False):
                            T.intel.tile.block_read(
                                "intel_sub_group_block_read8",
                                wr.view(1, 1, 8, 16, layout=w_read),
                                Wt[col0 // 16, sp * steps + s, 0:128],
                                scope="subgroup",
                            )
                            kk = (sp * steps + s) * STEP
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_8r16x2c",
                                ar.view(MR, 4, 8, 16, layout=a_read),
                                X[0:TM, kk : kk + STEP],
                                scope="subgroup",
                            )
                            scale[0] = Sc[kk // GROUP, col0 + lane]
                            # VNNI word p of K block b is the pair 16 b + 2 p, 16 b + 2 p + 1,
                            # whose nibbles sit at bits 4 j and 16 + 4 j of packed word
                            # 2 b + p // 4, j = p % 4.  OR-ing them under 0x6400 makes two
                            # float16 values 1024 + q + 8, so a half2 subtract and multiply
                            # give the pair scaled, already as the VNNI word.
                            for b in T.unroll(4):
                                for p in T.unroll(8):
                                    word = wr[b * 2 + p // 4]
                                    bits = (
                                        (word >> T.uint32(p % 4 * 4)) & T.uint32(0x000F000F)
                                    ) | T.uint32(0x64006400)
                                    pair = T.reinterpret(bits, ty="float16x2")
                                    value = (pair - T.Broadcast(T.float16(1032), 2)) * T.Broadcast(
                                        scale[0], 2
                                    )
                                    br[b * 8 + p] = T.reinterpret(value, ty="uint32")
                            T.intel.tile.dpas(
                                F16,
                                dr.view(MR, 1, 8, 16, layout=d),
                                ar.view(MR, 4, 8, 16, layout=a_dpas),
                                br.view(4, 1, 8, 16, layout=b_dpas),
                                dr.view(MR, 1, 8, 16, layout=d),
                                scope="subgroup",
                            )
                        if splits == 1:
                            T.intel.tile.block_write(
                                "intel_sub_group_2d_block_write_32b_8r16x1c",
                                Y[0:TM, col0 : col0 + 16],
                                dr.view(MR, 1, 8, 16, layout=d),
                                scope="subgroup",
                            )
                        else:
                            T.intel.tile.block_write(
                                "intel_sub_group_2d_block_write_32b_8r16x1c",
                                P[sp * TM : sp * TM + TM, col0 : col0 + 16],
                                dr.view(MR, 1, 8, 16, layout=d),
                                scope="subgroup",
                            )
            if splits > 1:
                with T.launch_thread(
                    "blockIdx.x", (m * n + reduce_threads - 1) // reduce_threads
                ) as rb:
                    with T.launch_thread("threadIdx.x", reduce_threads) as rt:
                        e = rb * reduce_threads + rt
                        if e < m * n:
                            acc = T.alloc_local((1,), "float32")
                            acc[0] = T.float32(0)
                            for j in range(splits):
                                acc[0] = acc[0] + P[j * TM + e // n, e % n]
                            Y[e // n, e % n] = acc[0]

    return Module


def _splits(m, n, k, splits=None):
    """How many work-groups share each output tile's K.

    Enough for about 256 work-groups when X is one fragment, fewer as it grows:
    each fragment adds DPAS work per weight read.  On an Arc Pro B60, 4096 x 4096
    ran fastest with 8 splits for one row of X and 2 for 32."""
    groups_n = n // (SUB_GROUPS * 16)
    if splits is None:
        splits = max(1, min(k // STEP, 256 // math.ceil(m / 8) // groups_n))
    splits = min(splits, k // STEP)
    while (k // STEP) % splits:
        splits -= 1
    return splits


def partials_shape(*, m, n, k, splits=None, **cfg):
    """The shape of ``main``'s partial sums ``P``."""
    return (_splits(m, n, k, splits) * math.ceil(m / 8) * 8, n)


# Where element 8 w + i of a row sits in word w: the even elements in the low nibble of each
# half's bytes, the odd ones 16 bits up, so a pair is one mask away from two float16 values.
_SHIFT = [4 * (i // 2) + 16 * (i % 2) for i in range(8)]


def quantize(w):
    """Symmetric int4 of float ``(n, k)`` weights: packed words and per-group scales.

    A value ``q`` in -8..7 is stored as the nibble ``q + 8``; element ``8 w + i``
    of a row sits in word ``w`` at bit ``4 (i // 2) + 16 (i % 2)``."""
    n, k = w.shape
    groups = w.astype(np.float32).reshape(n, k // GROUP, GROUP)
    scale = np.maximum(np.abs(groups).max(axis=-1, keepdims=True) / 7, 1e-8)
    q = np.clip(np.rint(groups / scale), -8, 7).astype(np.int32).reshape(n, k)
    nibbles = (q + 8).astype(np.uint32).reshape(n, k // 8, 8)
    packed = np.zeros((n, k // 8), np.uint32)
    for i in range(8):
        packed |= nibbles[:, :, i] << np.uint32(_SHIFT[i])
    return packed, scale[..., 0].astype(np.float16)


def pack(packed, scale):
    """Lay out :func:`quantize`'s words and scales for the kernel.

    Parameters
    ----------
    packed : numpy.ndarray
        ``(n, k // 8)`` uint32 words.
    scale : numpy.ndarray
        ``(n, k // 128)`` float16 scales.

    Returns
    -------
    tiled : numpy.ndarray
        ``(n // 16, k // 64, 128)`` uint32: for each 16 columns and 64 of K, word
        ``j`` of column ``i`` at ``16 j + i``.
    scales : numpy.ndarray
        ``(k // 128, n)`` float16.
    """
    n, words = packed.shape
    tiled = packed.reshape(n // 16, 16, words // 8, 8).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(tiled).reshape(n // 16, words // 8, 128), np.ascontiguousarray(
        scale.T
    )


def dequantize(packed, scale):
    """The float32 weights ``(n, k)`` that ``packed`` and ``scale`` stand for."""
    n, words = packed.shape
    nibbles = np.stack(
        [(packed >> np.uint32(_SHIFT[i])) & np.uint32(15) for i in range(8)], axis=-1
    )
    q = nibbles.astype(np.int32).reshape(n, words * 8) - 8
    return q.astype(np.float32) * np.repeat(scale.astype(np.float32), GROUP, axis=1)


def prepare_data(*, m, n, k, splits=None, **_):
    """Random X and weights, the weights quantized, a zeroed output and partials."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, k)).astype(np.float32) / math.sqrt(k)
    packed, scale = quantize(w)
    tiled, scales = pack(packed, scale)
    return {
        "X": rng.standard_normal((m, k)).astype(np.float16),
        "Wq": packed,
        "scale": scale,
        "Wt": tiled,
        "Sc": scales,
        "Y": np.zeros((m, n), np.float32),
        "P": np.zeros(partials_shape(m=m, n=n, k=k, splits=splits), np.float32),
    }


def check_correctness(outputs, **_):
    """Y against X times the dequantized weights in float32."""
    w = dequantize(outputs["Wq"], outputs["scale"]).astype(np.float16).astype(np.float32)
    want = outputs["X"].astype(np.float32) @ w.T
    np.testing.assert_allclose(outputs["Y"], want, rtol=2e-2, atol=2e-2)


def bytes_moved(*, m, n, k, **_):
    """Packed weights, scales, X and Y: the bytes the product must move once."""
    return n * k // 2 + n * (k // GROUP) * 2 + m * k * 2 + m * n * 4


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
        tensors = [tvm_tirx.runtime.tensor(data[key], dev) for key in ("X", "Wt", "Sc", "Y", "P")]
        fun["main"](*tensors)
        dev.sync()
        check_correctness(dict(data, Y=tensors[3].numpy()), **cfg)
        line = f"{cfg['label']:20s} correct"
        if args.bench:
            # A decode product takes tens of microseconds; time enough calls for the GPU's
            # clock to settle, or the first ones' ramp-up dominates.
            result = bench_interleaved({"tirx": lambda: fun["main"](*tensors)}, dev, number=200)
            us = result["tirx"].median_us
            line += f"  {us:7.1f} us  {bytes_moved(**cfg) / us / 1e3:5.0f} GB/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
