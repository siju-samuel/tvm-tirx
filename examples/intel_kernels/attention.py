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
"""FlashAttention prefill on Intel Xe DPAS: softmax(Q K^T / sqrt(d)) V, causal or not.

Each sub-group owns 16 query rows of one head and walks the keys 16 at a time,
keeping the output, the running row maximum and the running row sum in its
registers, so the score matrix is never stored.  For each block of keys:

* S = Q K^T with DPAS.  K is read as 32-bit words with the transposed 2D
  block read, which hands work-item n key n's head dimension in the VNNI
  pairs DPAS reads, so no repacking is needed.
* The block's row maximum, by ``T.intel.tile.reduce`` across the sub-group,
  updates the running one; the output and the row sum are rescaled by
  ``exp(old - new)``, and P = exp(S - max) by ``ewise``.
* P is converted to float16 in registers, its accumulator layout being that of
  an A operand, and O += P V with V read by the transforming block read.

At the end each row is divided by its sum.  With ``causal``, keys after the
query are masked on the diagonal block, and blocks wholly after it are skipped.

The module follows TIRx-kernels' kernel protocol.
"""

from __future__ import annotations

import math

import numpy as np

from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import R, S, TileLayout

KERNEL_META = {
    "name": "intel_flash_attention_prefill",
    "category": "attention",
    "runtime_targets": ["intel/arc-b60-opencl"],
}

CONFIGS = [
    {"label": "h8 l256 d64", "heads": 8, "length": 256, "head_dim": 64, "causal": False},
    {"label": "h8 l256 d64 causal", "heads": 8, "length": 256, "head_dim": 64, "causal": True},
    {"label": "h32 l2048 d64 causal", "heads": 32, "length": 2048, "head_dim": 64, "causal": True},
    {"label": "h32 l2048 d64", "heads": 32, "length": 2048, "head_dim": 64, "causal": False},
]

ROWS = 16  # query rows per sub-group: two 8-row DPAS fragments
KEYS = 16  # keys per step
SUB_GROUPS = 8  # sub-groups per work-group
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"
NEG = -1.0e30  # a masked score; exp(NEG - max) is 0


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


def _rows_layout(shape, strides):
    """One value per query row, on every work-item of the sub-group."""
    return TileLayout(S[tuple(shape) : tuple(strides)] + R[16 : 1 @ sglid])


@T.function(private=True)
def _exp_scaled(s: T.float32, m: T.float32, scale: T.float32) -> T.float32:
    return T.exp((s - m) * scale)


@T.function(private=True)
def _maximum(a: T.float32, b: T.float32) -> T.float32:
    return T.max(a, b)


@T.function(private=True)
def _rescale_sum(total: T.float32, alpha: T.float32, block: T.float32) -> T.float32:
    return total * alpha + block


@T.function(private=True)
def _times(x: T.float32, a: T.float32) -> T.float32:
    return x * a


@T.function(private=True)
def _divide(x: T.float32, d: T.float32) -> T.float32:
    return x / d


@T.function(private=True)
def _copy(x: T.float32) -> T.float32:
    return x


@T.function(private=True)
def _to_half(x: T.float32) -> T.float16:
    return T.Cast("float16", x)


@T.function(private=True)
def _mask(s: T.float32, row: T.int32, col: T.int32) -> T.float32:
    return T.if_then_else(col > row, T.float32(NEG), s)


def get_kernel(*, backend_config=None, heads, length, head_dim, causal, **_):
    """The attention as an IRModule whose ``main(Q, Kw, V, Out)`` launches one kernel.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    heads, length, head_dim : int
        Heads (batch times heads), sequence length and head dimension.  The
        length must be a multiple of 16 and the head dimension 64.
    causal : bool
        Mask keys after each query.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(Q, Kw, V, Out)``: Q and V are ``(heads, length, head_dim)``
        float16, Kw is K's ``(heads, length, head_dim // 2)`` float16 pairs as
        uint32 words, and Out is ``(heads, length, head_dim)`` float32.
    """
    if head_dim != 64:
        raise ValueError("this kernel handles head_dim 64")
    if length % (ROWS * SUB_GROUPS) != 0:
        raise ValueError(f"length must be a multiple of {ROWS * SUB_GROUPS}")
    D, KB = head_dim, head_dim // 16
    scale = 1.0 / math.sqrt(head_dim)
    steps = length // KEYS

    q_read = _lay((1, KB, 16, 16), (KB * 16, 16, 1))  # Q: register kb * 16 + row
    q_dpas = _lay((2, KB, 8, 16), (8, 16, 1))
    k_read = _lay((1, KB, 8, 16), (KB * 8, 8, 1))  # K^T: register kb * 8 + word
    k_dpas = _lay((KB, 1, 8, 16), (8, 8, 1))
    v_read = _lay((1, KB, 8, 16), (KB * 8, 8, 1))  # V: register nb * 8 + word
    v_dpas = _lay((1, KB, 8, 16), (KB * 8, 8, 1))
    s_lay = _lay((2, 1, 8, 16), (8, 8, 1))  # S and P: register frag * 8 + row
    o_lay = _lay((2, KB, 8, 16), (KB * 8, 8, 1))  # O: register frag * 32 + nb * 8 + row
    rows = _rows_layout((2, 1, 8, 1), (8, 8, 1, 1))
    rows_as_s = _rows_layout((2, 1, 8, 16), (8, 0, 1, 0))
    rows_as_o = _rows_layout((2, KB, 8, 16), (8, 0, 1, 0))
    cols_as_s = _lay((2, 1, 8, 16), (0, 0, 0))  # one key index per work-item

    @I.ir_module
    class Module:
        @T.function
        def main(
            Q: T.Tensor((heads, length, D), "float16"),
            Kw: T.Tensor((heads, length, D // 2), "uint32"),
            V: T.Tensor((heads, length, D), "float16"),
            Out: T.Tensor((heads, length, D), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.y", heads) as h:
                with T.launch_thread("blockIdx.x", length // (ROWS * SUB_GROUPS)) as qb:
                    with T.launch_thread("threadIdx.x", SUB_GROUPS * 16) as tx:
                        q = T.alloc_local((KB * 16,), "float16")
                        k = T.alloc_local((KB * 8,), "uint32")
                        v = T.alloc_local((KB * 8,), "uint32")
                        s = T.alloc_local((16,), "float32")
                        p = T.alloc_local((16,), "float16")
                        o = T.alloc_local((2 * KB * 8,), "float32")
                        m = T.alloc_local((16,), "float32")
                        m_new = T.alloc_local((16,), "float32")
                        total = T.alloc_local((16,), "float32")
                        part = T.alloc_local((16,), "float32")
                        row_id = T.alloc_local((16,), "int32")
                        col_id = T.alloc_local((1,), "int32")
                        lane = tx % 16
                        q0 = (qb * SUB_GROUPS + tx // 16) * ROWS
                        for e in range(16):
                            m[e] = T.float32(NEG)
                            total[e] = T.float32(0)
                            row_id[e] = q0 + e
                        for e in range(2 * KB * 8):
                            o[e] = T.float32(0)
                        T.intel.tile.block_read(
                            "intel_sub_group_2d_block_read_16b_16r16x1c",
                            q.view(1, KB, 16, 16, layout=q_read),
                            Q[h, q0 : q0 + ROWS, 0:D],
                            scope="subgroup",
                        )
                        for j in T.serial(q0 // KEYS + 1 if causal else steps, unroll=False):
                            kv0 = j * KEYS
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_transpose_32b_16r8x1c",
                                k.view(1, KB, 8, 16, layout=k_read),
                                Kw[h, kv0 : kv0 + KEYS, 0 : D // 2],
                                scope="subgroup",
                            )
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_transform_16b_16r16x1c",
                                v.view(1, KB, 8, 16, layout=v_read),
                                V[h, kv0 : kv0 + KEYS, 0:D],
                                scope="subgroup",
                            )
                            for e in range(16):
                                s[e] = T.float32(0)
                            T.intel.tile.dpas(
                                F16,
                                s.view(2, 1, 8, 16, layout=s_lay),
                                q.view(2, KB, 8, 16, layout=q_dpas),
                                k.view(KB, 1, 8, 16, layout=k_dpas),
                                s.view(2, 1, 8, 16, layout=s_lay),
                                scope="subgroup",
                            )
                            if causal:
                                col_id[0] = kv0 + lane
                                T.intel.tile.ewise(
                                    _mask,
                                    s.view(2, 1, 8, 16, layout=s_lay),
                                    s.view(2, 1, 8, 16, layout=s_lay),
                                    row_id.view(2, 1, 8, 16, layout=rows_as_s),
                                    col_id.view(2, 1, 8, 16, layout=cols_as_s),
                                )
                            # The running maximum, and the rescaling of what came before.
                            T.intel.tile.reduce(
                                "max",
                                part.view(2, 1, 8, 1, layout=rows),
                                s.view(2, 1, 8, 16, layout=s_lay),
                            )
                            T.intel.tile.ewise(
                                _maximum,
                                m_new.view(2, 1, 8, 1, layout=rows),
                                m.view(2, 1, 8, 1, layout=rows),
                                part.view(2, 1, 8, 1, layout=rows),
                            )
                            T.intel.tile.ewise(
                                _exp_scaled,
                                part.view(2, 1, 8, 1, layout=rows),
                                m.view(2, 1, 8, 1, layout=rows),
                                m_new.view(2, 1, 8, 1, layout=rows),
                                T.float32(scale),
                            )
                            T.intel.tile.ewise(
                                _times,
                                o.view(2, KB, 8, 16, layout=o_lay),
                                o.view(2, KB, 8, 16, layout=o_lay),
                                part.view(2, KB, 8, 16, layout=rows_as_o),
                            )
                            T.intel.tile.ewise(
                                _times,
                                total.view(2, 1, 8, 1, layout=rows),
                                total.view(2, 1, 8, 1, layout=rows),
                                part.view(2, 1, 8, 1, layout=rows),
                            )
                            T.intel.tile.ewise(
                                _copy,
                                m.view(2, 1, 8, 1, layout=rows),
                                m_new.view(2, 1, 8, 1, layout=rows),
                            )
                            # P = exp(S - max), its row sums, and O += P V.
                            T.intel.tile.ewise(
                                _exp_scaled,
                                s.view(2, 1, 8, 16, layout=s_lay),
                                s.view(2, 1, 8, 16, layout=s_lay),
                                m.view(2, 1, 8, 16, layout=rows_as_s),
                                T.float32(scale),
                            )
                            T.intel.tile.reduce(
                                "sum",
                                part.view(2, 1, 8, 1, layout=rows),
                                s.view(2, 1, 8, 16, layout=s_lay),
                            )
                            T.intel.tile.ewise(
                                _rescale_sum,
                                total.view(2, 1, 8, 1, layout=rows),
                                total.view(2, 1, 8, 1, layout=rows),
                                T.float32(1),
                                part.view(2, 1, 8, 1, layout=rows),
                            )
                            T.intel.tile.ewise(
                                _to_half,
                                p.view(2, 1, 8, 16, layout=s_lay),
                                s.view(2, 1, 8, 16, layout=s_lay),
                            )
                            T.intel.tile.dpas(
                                F16,
                                o.view(2, KB, 8, 16, layout=o_lay),
                                p.view(2, 1, 8, 16, layout=s_lay),
                                v.view(1, KB, 8, 16, layout=v_dpas),
                                o.view(2, KB, 8, 16, layout=o_lay),
                                scope="subgroup",
                            )
                        T.intel.tile.ewise(
                            _divide,
                            o.view(2, KB, 8, 16, layout=o_lay),
                            o.view(2, KB, 8, 16, layout=o_lay),
                            total.view(2, KB, 8, 16, layout=rows_as_o),
                        )
                        T.intel.tile.block_write(
                            "intel_sub_group_2d_block_write_32b_8r16x1c",
                            Out[h, q0 : q0 + ROWS, 0:D],
                            o.view(2, KB, 8, 16, layout=o_lay),
                            scope="subgroup",
                        )

    return Module


def prepare_data(*, heads, length, head_dim, **_):
    """Random Q, K and V, K also as uint32 words, and a zeroed output."""
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((heads, length, head_dim)).astype(np.float16) for _ in range(3))
    return {
        "Q": q,
        "K": k,
        "Kw": np.ascontiguousarray(k).view(np.uint32),
        "V": v,
        "O": np.zeros((heads, length, head_dim), np.float32),
    }


def reference(q, k, v, causal):
    """softmax(Q K^T / sqrt(d)) V in float32."""
    q, k, v = (x.astype(np.float32) for x in (q, k, v))
    s = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    s = np.exp(s - s.max(axis=-1, keepdims=True))
    return (s / s.sum(axis=-1, keepdims=True)) @ v


def check_correctness(outputs, *, causal, **_):
    """O against a float32 numpy attention of the same inputs."""
    want = reference(outputs["Q"], outputs["K"], outputs["V"], causal)
    np.testing.assert_allclose(outputs["O"], want, rtol=2e-2, atol=2e-3)


def flops(*, heads, length, head_dim, causal, **_):
    """The multiply-adds of Q K^T and P V, counted twice; half of them when causal."""
    total = 4 * heads * length * length * head_dim
    return total // 2 if causal else total


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
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ("Q", "Kw", "V", "O")]
        fun["main"](*tensors)
        dev.sync()
        check_correctness(dict(data, O=tensors[3].numpy()), **cfg)
        line = f"{cfg['label']:24s} correct"
        if args.bench:
            result = bench_interleaved({"tirx": lambda: fun["main"](*tensors)}, dev)
            line += f"  {flops(**cfg) / result['tirx'].median_us / 1e6:.1f} TFLOP/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
