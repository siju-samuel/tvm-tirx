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
"""Paged decode attention on Intel Xe DPAS: one new token per sequence over its KV cache.

The cache is paged, 16 tokens a page, and stored heads-major within a page,
``(pages, kv_heads, 16, head_dim)`` (FlashInfer's HND layout), so one page of one
head's keys or values is a plain 16-row matrix for the 2D block reads.
``table[b]`` lists sequence ``b``'s pages and ``lens[b]`` its length.  The query
heads that share a KV head, at most 8, are one 8-row DPAS fragment, so
grouped-query attention reads each page once for the whole group.

Each sub-group takes one sequence, one KV head and one share of the
sequence's pages, and walks them with the prefill kernel's online softmax:

* S = Q K^T, the keys through the transposed 32-bit read, which hands work-item
  n key n's head dimension already in the VNNI pairs DPAS reads.
* The values are read into the keys' registers once S is formed, so their load
  overlaps the softmax and a 128-wide head fits the register file; the next
  page is prefetched.
* Scores past the sequence's length are masked, the running row maximum and
  sum come from ``T.intel.tile.reduce``, and O += P V with the values through
  the transforming read.

Fragment rows past the group read as zeros and are not written: the queries
and outputs are addressed through views whose height ends at the group's last
row, so the block reads and writes mask them in hardware.  The shares of one
work-group merge in shared local memory; those of separate work-groups each
write their unnormalized output and their rows' maximum and sum, which a second
kernel merges (FlashDecoding).  :func:`plan` chooses the split.

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
    "name": "intel_paged_decode_attention",
    "category": "attention",
    "runtime_targets": ["intel/arc-b60-opencl"],
}

CONFIGS = [
    {
        "label": "b4 h32/8 l300",
        "batch": 4,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "max_len": 300,
    },
    {
        "label": "b32 h32/8 l2048",
        "batch": 32,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "max_len": 2048,
    },
    {
        "label": "b8 h32/8 l8192",
        "batch": 8,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "max_len": 8192,
    },
    {
        "label": "b16 h64/8 l4096",
        "batch": 16,
        "heads": 64,
        "kv_heads": 8,
        "head_dim": 128,
        "max_len": 4096,
    },
]

PAGE = 16  # tokens per page
F16 = "intel_sub_group_f16_f16_matrix_mad_k16"
NEG = -1.0e30  # a masked score; exp(NEG - max) is 0
LOG2E = 1.4426950408889634
MERGE_THREADS = 256


def _lay(shape, strides):
    return TileLayout(S[tuple(shape) : (*strides, 1 @ sglid)])


def _rows(shape, strides):
    """One value per query row, on every work-item of the sub-group."""
    return TileLayout(S[tuple(shape) : tuple(strides)] + R[16 : 1 @ sglid])


@T.function(private=True)
def _exp_scaled(s: T.float32, m: T.float32, c: T.float32) -> T.float32:
    # exp((s - m) * scale) as exp2 with c = scale * log2(e), which IGC compiles to the math
    # unit's 2^x rather than exp's dozen instructions.  Subtracting first keeps an empty
    # share's maximum minus itself at 0, which s * c - m * c, fused into a multiply-add, does not.
    return T.exp2((s - m) * c)


@T.function(private=True)
def _maximum(a: T.float32, b: T.float32) -> T.float32:
    return T.max(a, b)


@T.function(private=True)
def _times(x: T.float32, a: T.float32) -> T.float32:
    return x * a


@T.function(private=True)
def _plus(x: T.float32, y: T.float32) -> T.float32:
    return x + y


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
def _mask(s: T.float32, token: T.int32, length: T.int32) -> T.float32:
    return T.if_then_else(token >= length, T.float32(NEG), s)


def plan(*, batch, kv_heads, head_dim, max_len, splits=None, sub_groups=None, **_):
    """How each sequence's pages are shared: ``(splits, sub_groups)``.

    ``splits`` work-groups take each sequence and KV head, each with
    ``sub_groups`` sub-groups, one share of the pages apiece.  The default is
    256 shares across the launch, at least 2 pages each, in separate one-sub-group
    work-groups: on an Arc Pro B60 that beat packing the same shares into fewer,
    larger work-groups by up to 28% whenever the shares were long.  When they are
    at most 4 pages and there are at least 16 sequence-head pairs, the shares
    instead stay in one work-group and merge in shared local memory, which saves
    the merge kernel's launch: 12.8 instead of 17 us for 4 sequences of 300
    tokens.  A work-group's shares are 8 rows of ``head_dim`` floats each, at
    most 64 KiB of them.
    """
    pairs = batch * kv_heads
    pages = math.ceil(max_len / PAGE)
    cap = max(1, min(16, 65536 // (8 * head_dim * 4)))
    shares = max(1, min(256 // pairs, pages // 2))
    together = pairs >= 16 and math.ceil(pages / shares) <= 4
    if splits is None:
        splits = math.ceil(shares / cap) if together else shares
    if sub_groups is None:
        sub_groups = max(1, min(cap, math.ceil(shares / splits)))
    return splits, sub_groups


def get_kernel(
    *,
    backend_config=None,
    batch,
    heads,
    kv_heads,
    head_dim,
    max_len,
    splits=None,
    sub_groups=None,
    **_,
):
    """Decode attention as an IRModule whose ``main(Q, Kw, V, table, lens, Out, Op, Ms)`` runs.

    Parameters
    ----------
    backend_config : dict, optional
        Unused; accepted for the kernel protocol.
    batch, heads, kv_heads, head_dim, max_len : int
        Sequences, query and KV heads, the head dimension (a multiple of 16), and
        the longest sequence.  ``heads`` must be ``kv_heads`` times at most 8.
    splits, sub_groups : int, optional
        Work-groups sharing each sequence's pages, and sub-groups in each;
        :func:`plan` by default.

    Returns
    -------
    module : tvm_tirx.IRModule
        ``main(Q, Kw, V, table, lens, Out, Op, Ms)``.  ``Q`` is the ``(batch * heads,
        head_dim)`` float16 queries, query head ``h`` of sequence ``b`` in row ``b *
        heads + h``.  ``Kw`` is the key pages, ``(pages, kv_heads, 16, head_dim)``
        float16, as ``(pages, kv_heads, 16, head_dim // 2)`` uint32 words; ``V`` the
        value pages, float16; any number of pages.  ``table`` is the ``(batch,
        ceil(max_len / 16))`` int32 page table and ``lens`` the ``(batch,)`` int32
        lengths, each at least 1.  ``Out`` is the ``(batch * heads, head_dim)``
        float32 output, and ``Op`` and ``Ms`` the shares' outputs and row
        statistics, of :func:`workspace_shapes`.
    """
    group = heads // kv_heads
    if group * kv_heads != heads or group > 8:
        raise ValueError("heads must be kv_heads times at most 8")
    if head_dim % 16:
        raise ValueError("head_dim must be a multiple of 16")
    D, KB = head_dim, head_dim // 16
    splits, sub_groups = plan(
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_len=max_len,
        splits=splits,
        sub_groups=sub_groups,
    )
    shares = splits * sub_groups
    max_pages = math.ceil(max_len / PAGE)
    rows_total = batch * heads
    c = LOG2E / math.sqrt(head_dim)  # the softmax scale, for exp2
    pool = T.dynamic("pool", "int32")
    # Q, the keys, the values and O by 16-wide block of the head dimension:
    # register block * 8 + row for Q and O, block * 8 + VNNI word for K and V.
    by_block = _lay((1, KB, 8, 16), (KB * 8, 8, 1))
    k_dpas = _lay((KB, 1, 8, 16), (8, 8, 1))
    s_lay = _lay((1, 1, 8, 16), (8, 8, 1))  # S and P: register row, work-item key
    rows = _rows((1, 1, 8, 1), (8, 8, 1, 1))
    rows_as_s = _rows((1, 1, 8, 16), (8, 0, 1, 0))
    rows_as_o = _rows((1, KB, 8, 16), (8, 0, 1, 0))
    token_as_s = _lay((1, 1, 8, 16), (0, 0, 0))  # each work-item's key
    length_as_s = _rows((1, 1, 8, 16), (0, 0, 0, 0))
    prefetch = "intel_sub_group_2d_block_prefetch_16b_16r16x2c"

    @I.ir_module
    class Module:
        @T.function
        def main(
            Q: T.Tensor((rows_total, D), "float16"),
            Kw: T.Tensor((pool, kv_heads, PAGE, D // 2), "uint32"),
            V: T.Tensor((pool, kv_heads, PAGE, D), "float16"),
            table: T.Tensor((batch, max_pages), "int32"),
            lens: T.Tensor((batch,), "int32"),
            Out: T.Tensor((rows_total, D), "float32"),
            Op: T.Tensor((splits * rows_total, D), "float32"),
            Ms: T.Tensor((splits * rows_total, 2), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.z", batch) as b:
                with T.launch_thread("blockIdx.y", kv_heads) as kh:
                    with T.launch_thread("blockIdx.x", splits) as sp:
                        with T.launch_thread("threadIdx.x", sub_groups * 16) as tx:
                            sg = tx // 16
                            lane = tx % 16
                            q = T.alloc_local((KB * 8,), "float16")
                            kv = T.alloc_local((KB * 8,), "uint32")
                            s = T.alloc_local((8,), "float32")
                            p = T.alloc_local((8,), "float16")
                            o = T.alloc_local((KB * 8,), "float32")
                            m = T.alloc_local((8,), "float32")
                            m_new = T.alloc_local((8,), "float32")
                            total = T.alloc_local((8,), "float32")
                            part = T.alloc_local((8,), "float32")
                            token = T.alloc_local((1,), "int32")
                            length = T.alloc_local((1,), "int32")
                            row0 = b * heads + kh * group
                            # Heights ending at the group's last head: the rows past it
                            # read as zeros and are not written.
                            Qg = T.decl_tensor(
                                Q.data,
                                ty_args=[T.Tensor((row0 + group, D), "float16", data=Q.data)],
                            )
                            # The key pages as halves, for the 16-bit prefetch.
                            Kh = T.decl_tensor(
                                Kw.data,
                                ty_args=[
                                    T.Tensor((pool, kv_heads, PAGE, D), "float16", data=Kw.data)
                                ],
                            )
                            # From lens itself, not the register copy, so the page loop's
                            # bound is provably the same across the sub-group.
                            length[0] = lens[b]
                            pages = (lens[b] + PAGE - 1) // PAGE
                            share = (pages + shares - 1) // shares
                            first = (sp * sub_groups + sg) * share
                            count = T.max(T.min(pages, first + share) - first, 0)
                            for e in range(8):
                                m[e] = T.float32(NEG)
                                total[e] = T.float32(0)
                            for e in range(KB * 8):
                                o[e] = T.float32(0)
                            T.intel.tile.block_read(
                                "intel_sub_group_2d_block_read_16b_8r16x1c",
                                q.view(1, KB, 8, 16, layout=by_block),
                                Qg[row0 : row0 + 8, 0:D],
                                scope="subgroup",
                            )
                            for j in T.serial(count, unroll=False):
                                page = table[b, first + j]
                                after = table[b, first + T.min(j + 1, count - 1)]
                                T.intel.tile.block_prefetch(
                                    prefetch, Kh[after, kh, 0:PAGE, 0:D], scope="subgroup"
                                )
                                T.intel.tile.block_prefetch(
                                    prefetch, V[after, kh, 0:PAGE, 0:D], scope="subgroup"
                                )
                                T.intel.tile.block_read(
                                    "intel_sub_group_2d_block_read_transpose_32b_16r8x1c",
                                    kv.view(1, KB, 8, 16, layout=by_block),
                                    Kw[page, kh, 0:PAGE, 0 : D // 2],
                                    scope="subgroup",
                                )
                                for e in range(8):
                                    s[e] = T.float32(0)
                                T.intel.tile.dpas(
                                    F16,
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    q.view(1, KB, 8, 16, layout=by_block),
                                    kv.view(KB, 1, 8, 16, layout=k_dpas),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    scope="subgroup",
                                )
                                T.intel.tile.block_read(
                                    "intel_sub_group_2d_block_read_transform_16b_16r16x1c",
                                    kv.view(1, KB, 8, 16, layout=by_block),
                                    V[page, kh, 0:PAGE, 0:D],
                                    scope="subgroup",
                                )
                                token[0] = (first + j) * PAGE + lane
                                T.intel.tile.ewise(
                                    _mask,
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    token.view(1, 1, 8, 16, layout=token_as_s),
                                    length.view(1, 1, 8, 16, layout=length_as_s),
                                )
                                # The running maximum, and the rescaling of what came before.
                                T.intel.tile.reduce(
                                    "max",
                                    part.view(1, 1, 8, 1, layout=rows),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                )
                                T.intel.tile.ewise(
                                    _maximum,
                                    m_new.view(1, 1, 8, 1, layout=rows),
                                    m.view(1, 1, 8, 1, layout=rows),
                                    part.view(1, 1, 8, 1, layout=rows),
                                )
                                T.intel.tile.ewise(
                                    _exp_scaled,
                                    part.view(1, 1, 8, 1, layout=rows),
                                    m.view(1, 1, 8, 1, layout=rows),
                                    m_new.view(1, 1, 8, 1, layout=rows),
                                    T.float32(c),
                                )
                                T.intel.tile.ewise(
                                    _times,
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    part.view(1, KB, 8, 16, layout=rows_as_o),
                                )
                                T.intel.tile.ewise(
                                    _times,
                                    total.view(1, 1, 8, 1, layout=rows),
                                    total.view(1, 1, 8, 1, layout=rows),
                                    part.view(1, 1, 8, 1, layout=rows),
                                )
                                T.intel.tile.ewise(
                                    _copy,
                                    m.view(1, 1, 8, 1, layout=rows),
                                    m_new.view(1, 1, 8, 1, layout=rows),
                                )
                                # P = exp(S - max), its row sums, and O += P V.
                                T.intel.tile.ewise(
                                    _exp_scaled,
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                    m.view(1, 1, 8, 16, layout=rows_as_s),
                                    T.float32(c),
                                )
                                T.intel.tile.reduce(
                                    "sum",
                                    part.view(1, 1, 8, 1, layout=rows),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                )
                                T.intel.tile.ewise(
                                    _plus,
                                    total.view(1, 1, 8, 1, layout=rows),
                                    total.view(1, 1, 8, 1, layout=rows),
                                    part.view(1, 1, 8, 1, layout=rows),
                                )
                                T.intel.tile.ewise(
                                    _to_half,
                                    p.view(1, 1, 8, 16, layout=s_lay),
                                    s.view(1, 1, 8, 16, layout=s_lay),
                                )
                                T.intel.tile.dpas(
                                    F16,
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    p.view(1, 1, 8, 16, layout=s_lay),
                                    kv.view(1, KB, 8, 16, layout=by_block),
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    scope="subgroup",
                                )
                            prow = sp * rows_total + row0
                            if sub_groups == 1 and splits == 1:
                                T.intel.tile.ewise(
                                    _divide,
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    total.view(1, KB, 8, 16, layout=rows_as_o),
                                )
                                Og = T.decl_tensor(
                                    Out.data,
                                    ty_args=[T.Tensor((row0 + group, D), "float32", data=Out.data)],
                                )
                                T.intel.tile.block_write(
                                    "intel_sub_group_2d_block_write_32b_8r16x1c",
                                    Og[row0 : row0 + 8, 0:D],
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    scope="subgroup",
                                )
                            elif sub_groups == 1:
                                # This share's unnormalized output, and its rows' maximum and sum.
                                Opg = T.decl_tensor(
                                    Op.data,
                                    ty_args=[T.Tensor((prow + group, D), "float32", data=Op.data)],
                                )
                                T.intel.tile.block_write(
                                    "intel_sub_group_2d_block_write_32b_8r16x1c",
                                    Opg[prow : prow + 8, 0:D],
                                    o.view(1, KB, 8, 16, layout=by_block),
                                    scope="subgroup",
                                )
                                if lane == 0:
                                    for g in T.unroll(group):
                                        Ms[prow + g, 0] = m[g]
                                        Ms[prow + g, 1] = total[g]
                            else:
                                # The work-group's shares meet in shared local memory.
                                part_o = T.alloc_shared((sub_groups, 8, D), "float32")
                                part_m = T.alloc_shared((sub_groups, 8), "float32")
                                part_t = T.alloc_shared((sub_groups, 8), "float32")
                                for kb in T.unroll(KB):
                                    for r in T.unroll(8):
                                        part_o[sg, r, kb * 16 + lane] = o[kb * 8 + r]
                                # Every work-item of a sub-group holds the same statistics.
                                for r in T.unroll(8):
                                    part_m[sg, r] = m[r]
                                    part_t[sg, r] = total[r]
                                T.gpu_storage_sync("shared")
                                best = T.alloc_local((1,), "float32")
                                num = T.alloc_local((1,), "float32")
                                den = T.alloc_local((1,), "float32")
                                for i in T.serial(-(-group * D // (sub_groups * 16))):
                                    e = i * sub_groups * 16 + tx
                                    if e < group * D:
                                        r = e // D
                                        best[0] = T.float32(NEG)
                                        for j in range(sub_groups):
                                            best[0] = T.max(best[0], part_m[j, r])
                                        num[0] = T.float32(0)
                                        den[0] = T.float32(0)
                                        for j in range(sub_groups):
                                            w = T.exp2((part_m[j, r] - best[0]) * c)
                                            num[0] = num[0] + part_o[j, r, e % D] * w
                                            den[0] = den[0] + part_t[j, r] * w
                                        if splits == 1:
                                            Out[row0 + r, e % D] = num[0] / den[0]
                                        else:
                                            Op[prow + r, e % D] = num[0]
                                            if e % D == 0:
                                                Ms[prow + r, 0] = best[0]
                                                Ms[prow + r, 1] = den[0]
            if splits > 1:
                # Each share's output and sum, scaled by exp(its maximum - the overall one).
                with T.launch_thread(
                    "blockIdx.x", (rows_total * D + MERGE_THREADS - 1) // MERGE_THREADS
                ) as mb:
                    with T.launch_thread("threadIdx.x", MERGE_THREADS) as mt:
                        e = mb * MERGE_THREADS + mt
                        if e < rows_total * D:
                            r = e // D
                            best = T.alloc_local((1,), "float32")
                            num = T.alloc_local((1,), "float32")
                            den = T.alloc_local((1,), "float32")
                            best[0] = T.float32(NEG)
                            for j in range(splits):
                                best[0] = T.max(best[0], Ms[j * rows_total + r, 0])
                            num[0] = T.float32(0)
                            den[0] = T.float32(0)
                            for j in range(splits):
                                w = T.exp2((Ms[j * rows_total + r, 0] - best[0]) * c)
                                num[0] = num[0] + Op[j * rows_total + r, e % D] * w
                                den[0] = den[0] + Ms[j * rows_total + r, 1] * w
                            Out[r, e % D] = num[0] / den[0]

    return Module


def workspace_shapes(
    *, batch, heads, head_dim, kv_heads, max_len, splits=None, sub_groups=None, **_
):
    """The shapes of ``main``'s ``Op`` and ``Ms``, unused with one work-group a sequence."""
    splits, _ = plan(
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_len=max_len,
        splits=splits,
        sub_groups=sub_groups,
    )
    rows = splits * batch * heads
    return (rows, head_dim), (rows, 2)


def prepare_data(
    *,
    batch,
    heads,
    kv_heads,
    head_dim,
    max_len,
    splits=None,
    sub_groups=None,
    lengths="random",
    seed=0,
    **_,
):
    """Random queries and a paged cache whose pages are scattered through a shuffled pool.

    ``lengths`` is ``"random"`` (between 1 and ``max_len``, the first sequence at
    ``max_len``), ``"full"`` (every sequence at ``max_len``) or a list.
    """
    rng = np.random.default_rng(seed)
    if lengths == "random":
        lengths = rng.integers(1, max_len + 1, batch)
        lengths[0] = max_len
    elif lengths == "full":
        lengths = [max_len] * batch
    lengths = np.asarray(lengths, np.int32)
    used = [math.ceil(n / PAGE) for n in lengths]
    pool = sum(used) + 1  # and one page no sequence owns
    order = rng.permutation(pool).astype(np.int32)
    table = np.zeros((batch, math.ceil(max_len / PAGE)), np.int32)
    start = 0
    for b, n in enumerate(used):
        table[b, :n] = order[start : start + n]
        start += n
    q = rng.standard_normal((batch * heads, head_dim)).astype(np.float16)
    k, v = (
        rng.standard_normal((pool, kv_heads, PAGE, head_dim)).astype(np.float16) for _ in range(2)
    )
    op, ms = workspace_shapes(
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        kv_heads=kv_heads,
        max_len=max_len,
        splits=splits,
        sub_groups=sub_groups,
    )
    return {
        "Q": q,
        "K": k,
        "Kw": k.view(np.uint32),
        "V": v,
        "table": table,
        "lens": lengths,
        "O": np.zeros((batch * heads, head_dim), np.float32),
        # NaN, so a merge that read what no share wrote would show.
        "Op": np.full(op, np.nan, np.float32),
        "Ms": np.full(ms, np.nan, np.float32),
    }


def reference(q, k, v, table, lens, *, heads, kv_heads):
    """softmax(q K^T / sqrt(d)) V for each sequence's cached tokens, in float32."""
    group, D = heads // kv_heads, q.shape[1]
    out = np.zeros(q.shape, np.float32)
    for b, n in enumerate(lens):
        pages = table[b, : math.ceil(n / PAGE)]
        keys, values = (
            x[pages].astype(np.float32).transpose(1, 0, 2, 3).reshape(kv_heads, -1, D)[:, :n]
            for x in (k, v)
        )
        rows = q[b * heads : (b + 1) * heads].astype(np.float32).reshape(kv_heads, group, D)
        s = rows @ keys.transpose(0, 2, 1) / math.sqrt(D)
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        out[b * heads : (b + 1) * heads] = ((s / s.sum(axis=-1, keepdims=True)) @ values).reshape(
            heads, D
        )
    return out


def check_correctness(outputs, *, heads, kv_heads, **_):
    """O against a float32 numpy attention of the same inputs."""
    want = reference(
        outputs["Q"],
        outputs["K"],
        outputs["V"],
        outputs["table"],
        outputs["lens"],
        heads=heads,
        kv_heads=kv_heads,
    )
    np.testing.assert_allclose(outputs["O"], want, rtol=2e-2, atol=2e-3)


def bytes_moved(*, batch, heads, kv_heads, head_dim, max_len, **_):
    """The keys and values of ``batch`` sequences of ``max_len``, and the queries and outputs."""
    return 4 * batch * kv_heads * max_len * head_dim + 6 * batch * heads * head_dim


ARGS = ("Q", "Kw", "V", "table", "lens", "O", "Op", "Ms")


def main(argv=None):
    """Check every config, or the named ones, and with ``--bench`` time them."""
    import argparse  # pylint: disable=import-outside-toplevel

    import tvm_tirx  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.intel.target_tags import (
        device_tag,  # pylint: disable=import-outside-toplevel
    )
    from tvm_tirx.testing.device_bench import (  # pylint: disable=import-outside-toplevel
        bench_interleaved,
        separator,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="*", help="config labels; all by default")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="time each config at full lengths, a small kernel after each call",
    )
    parser.add_argument("--splits", type=int, default=None, help="work-groups per sequence")
    parser.add_argument("--sub-groups", type=int, default=None, help="sub-groups per work-group")
    parser.add_argument("--device", type=int, default=0, help="OpenCL device index")
    args = parser.parse_args(argv)
    dev = tvm_tirx.opencl(args.device)
    target = device_tag(args.device) or KERNEL_META["runtime_targets"][0]
    between = separator(dev, target)
    for cfg in CONFIGS:
        if args.labels and cfg["label"] not in args.labels:
            continue
        cfg = dict(cfg, splits=args.splits, sub_groups=args.sub_groups)
        data = prepare_data(**cfg, lengths="full" if args.bench else "random")
        fun = tvm_tirx.compile(get_kernel(**cfg), target=target)
        tensors = [tvm_tirx.runtime.tensor(data[k], dev) for k in ARGS]
        fun["main"](*tensors)
        dev.sync()
        check_correctness(dict(data, O=tensors[5].numpy()), **cfg)
        line = f"{cfg['label']:18s} shares {'x'.join(map(str, plan(**cfg)))}  correct"
        if args.bench:
            # Back to back, one launch's tail overlaps the next, as it cannot in a model.
            result = bench_interleaved(
                {"tirx": lambda: fun["main"](*tensors)},
                dev,
                number=200,
                between=between,
            )
            us = result["tirx"].median_us
            line += f"  {us:7.1f} us  {bytes_moved(**cfg) / us / 1e3:6.1f} GB/s"
        print(line, flush=True)


if __name__ == "__main__":
    main()
