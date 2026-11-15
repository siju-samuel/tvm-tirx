..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Intel Xe Matrix Instructions
============================

Intel Xe GPUs multiply matrices with XMX engines, which OpenCL exposes as
``intel_sub_group_*_matrix_mad_*`` builtins (``cl_intel_subgroup_matrix_multiply_accumulate``).
``T.opencl.tile.dpas`` issues one of them over fragments held in the registers
of a 16-work-item sub-group, and checks at compile time that each fragment is
laid out the way the hardware reads it. This guide multiplies an ``8 x 16``
float16 matrix by a ``16 x 16`` one with float32 accumulation on an Arc Pro B60.

Fragments and the Sub-group Lane Axis
-------------------------------------

Every operand of the instruction is a register fragment: part of a matrix
spread across the 16 work-items of a sub-group. A register layout says which
work-item holds each value with the ``sglid`` axis, the work-item's index within
its sub-group (``get_sub_group_local_id()``).

All three operands use the same mapping. Atom coordinate ``(row, col)`` lives in
work-item ``col``, at register component ``row``:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Operand
     - Atom
     - Storage word
   * - ``D``, ``C``
     - ``(M, 16)``
     - One 32-bit accumulator element: float32, float16, or int32.
   * - ``A``
     - ``(M, 16)``
     - 16 bits: one float16 or bfloat16 element, or two int8 elements
       adjacent in K.
   * - ``B``
     - ``(8, 16)``
     - 32 bits: the K-adjacent elements that fill a word, two 16-bit or four
       8-bit. Word ``q`` of column ``n`` packs rows ``2q`` and ``2q + 1``.

``M`` is 1, 2, 4 or 8. As in the CUDA tile layer, the dtype describes the
storage word and the instruction does not pack values, so a float16 ``B``
fragment is passed as a ``uint32`` view. Because packed words do not carry
their element type, the builtin is named explicitly.

A Matrix Product in One Sub-group
---------------------------------

.. code-block:: python

    from tvm_tirx.backend.opencl.axis import sglid
    from tvm_tirx.script import ir as I
    from tvm_tirx.script import tirx as T
    from tvm_tirx.tirx.layout import S, TileLayout

    fragment = TileLayout(S[(1, 1, 8, 16) : (8, 8, 1, 1 @ sglid)])


    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((8, 16), "float16"),
            B: T.Tensor((16, 16), "float16"),
            C: T.Tensor((8, 16), "float32"),
            D: T.Tensor((8, 16), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1):
                with T.launch_thread("threadIdx.x", 16) as lane:
                    ar = T.alloc_local((8,), "float16")
                    br = T.alloc_local((16,), "float16")
                    cr = T.alloc_local((8,), "float32")
                    for m in range(8):
                        ar[m] = A[m, lane]
                        cr[m] = C[m, lane]
                    for k in range(16):
                        br[k] = B[k, lane]
                    T.opencl.tile.dpas(
                        "intel_sub_group_f16_f16_matrix_mad_k16",
                        cr.view(1, 1, 8, 16, layout=fragment),
                        ar.view(1, 1, 8, 16, layout=fragment),
                        br.view(T.uint32.dtype).view(1, 1, 8, 16, layout=fragment),
                        cr.view(1, 1, 8, 16, layout=fragment),
                        scope="subgroup",
                    )
                    for m in range(8):
                        D[m, lane] = cr[m]

Each work-item loads column ``lane`` of every operand: eight rows of ``A`` and
``C``, and all sixteen K rows of ``B``. The ``uint32`` view of ``br`` packs
consecutive K pairs into the words the instruction reads. The instruction takes
``D, A, B, C``; passing ``cr`` as both ``D`` and ``C`` accumulates in place.

The kernel compiles for an Intel target and runs through the OpenCL runtime:

.. code-block:: python

    import numpy as np
    import tvm_tirx

    compiled = tvm_tirx.compile(Module, target="intel/arc-b60-opencl")
    a = np.random.uniform(-1, 1, (8, 16)).astype("float16")
    b = np.random.uniform(-1, 1, (16, 16)).astype("float16")
    c = np.random.uniform(-1, 1, (8, 16)).astype("float32")
    dev = tvm_tirx.opencl(0)
    args = [tvm_tirx.runtime.tensor(x, dev) for x in (a, b, c, np.zeros((8, 16), "float32"))]
    compiled["main"](*args)
    np.testing.assert_allclose(args[3].numpy(), a.astype("float32") @ b.astype("float32") + c,
                               rtol=1e-3, atol=1e-3)

Larger Tiles and K Accumulation
-------------------------------

Leading axes before each atom repeat the instruction:

.. code-block:: text

    D, C: (*batch, Mr, Nr, M, 16)
    A:    (*batch, Mr, Kr, M, 16)
    B:    (*batch, Kr, Nr, 8, 16)

Each ``(Mr, Nr)`` position issues ``Kr`` instructions. The first reads ``C`` and
the rest accumulate into the preceding ``D``. ``C`` may also be a scalar, such
as ``T.float32(0)``, broadcast across the accumulator. A sub-group computing a
``32 x 32`` output tile with ``Mr = 4`` and ``Nr = 2`` passes

.. code-block:: python

    def fragment(r0, r1, rows):
        return TileLayout(S[(r0, r1, rows, 16) : (r1 * rows, rows, 1, 1 @ sglid)])

    T.opencl.tile.dpas(
        "intel_sub_group_f16_f16_matrix_mad_k16",
        dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
        ar.view(4, 1, 8, 16, layout=fragment(4, 1, 8)),
        br.view(T.uint32.dtype).view(1, 2, 8, 16, layout=fragment(1, 2, 8)),
        dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
        scope="subgroup",
    )

and emits eight instructions. The repeat counts are static; each repeat becomes
straight-line code.

Instructions
------------

.. list-table::
   :header-rows: 1
   :widths: 50 20 30

   * - Builtin
     - K per instruction
     - Accumulator
   * - ``intel_sub_group_f16_f16_matrix_mad_k16``
     - 16
     - float32 or float16
   * - ``intel_sub_group_bf16_bf16_matrix_mad_k16``
     - 16
     - float32 or bfloat16
   * - ``intel_sub_group_tf32_tf32_matrix_mad_k8``
     - 8
     - float32
   * - ``intel_sub_group_i8_i8_matrix_mad_k32``, ``u8_u8``, ``u8_i8``, ``i8_u8``
     - 32
     - int32

The unsigned families take unsigned vectors, and the lowering reinterprets each
operand to the exact types the builtin declares. A bfloat16 accumulator travels
through the builtin as raw 16-bit words. tf32 keeps two A rows in each register
component, so its A atom is ``(M / 2, 2, 8)`` rather than ``(M, 16)``:
coordinate ``(c, h, k)`` is row ``2c + h``, held by work-item ``8h + k``. Its
layout is therefore ``S[(..., M // 2, 2, 8) : (..., 1, 8 @ sglid, 1 @ sglid)]``,
and an A laid out the way other families expect is rejected.

What Is Checked
---------------

Lowering proves, for each repeat and over every work-item and component of the
atom, that the operand's layout places the value on ``sglid == col`` and that a
work-item's components sit in consecutive registers. It also checks that output
repeats do not share registers, that ``A`` and ``B`` use 16- and 32-bit words,
that the accumulator type suits the builtin, and that the target is an Intel
part with a declared sub-group size of 16. A fragment that disagrees with the
hardware fails to compile with the reason, instead of returning wrong numbers.

Like every sub-group collective, the instruction requires all 16 work-items of
the sub-group to reach it, and ``TileDispatch`` proves that before lowering, as
it does for CUDA warp instructions. The work-group's x extent must be a multiple
of 16, and nothing on the way to the call may treat work-items of one sub-group
differently: not an enclosing condition, in either branch, nor a loop bound, nor
a ``break``, ``continue`` or ``return``. A condition is proved uniform rather
than recognized, so ``if tx // 16 < 2`` is accepted and ``if tx % 16 < 8`` is
rejected with the condition named. A condition that reads private memory counts
as divergent even at a uniform index, since each work-item reads its own
register.

The check covers the instruction, not scalar code that fills the registers.
The loads in the example index by ``lane``, and the lowering takes on trust that
the loaded values match the layout; the block reads below are checked end to
end.

Loading Fragments with 2D Block Reads
-------------------------------------

Scalar loads are enough to check correctness but leave the XMX engines
underfed: a ``32 x 32`` tile issues 64 loads per work-item for every 8 matrix
instructions. The 2D block reads of ``cl_intel_subgroup_2d_block_io`` move a
whole block per instruction, and they produce exactly the fragments DPAS reads,
so the same register views pass from one instruction to the next.

``T.opencl.tile.block_read(instruction, dst, src, scope="subgroup")`` reads a
region of a row-major global matrix. The region is tiled into the instruction's
blocks, and the register operand has one atom per block, with repeat axes over
the row and column blocks. ``block_write`` stores registers back, with the
global region first. A sub-group's main loop then reads its ``A`` rows and
``B`` columns, multiplies, and finally writes its ``C`` tile:

.. code-block:: python

    for kb in range(K // 16):
        T.opencl.tile.block_read(
            "intel_sub_group_2d_block_read_16b_8r16x1c",
            ar.view(4, 1, 8, 16, layout=fragment(4, 1, 8)),
            A[row0 : row0 + 32, kb * 16 : kb * 16 + 16],
            scope="subgroup",
        )
        T.opencl.tile.block_read(
            "intel_sub_group_2d_block_read_transform_16b_16r16x1c",
            br.view(T.uint32.dtype).view(1, 2, 8, 16, layout=fragment(1, 2, 8)),
            B[kb * 16 : kb * 16 + 16, col0 : col0 + 32],
            scope="subgroup",
        )
        T.opencl.tile.dpas(
            "intel_sub_group_f16_f16_matrix_mad_k16",
            dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
            ar.view(4, 1, 8, 16, layout=fragment(4, 1, 8)),
            br.view(T.uint32.dtype).view(1, 2, 8, 16, layout=fragment(1, 2, 8)),
            dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
            scope="subgroup",
        )
    T.opencl.tile.block_write(
        "intel_sub_group_2d_block_write_32b_8r16x1c",
        C[row0 : row0 + 32, col0 : col0 + 32],
        dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
        scope="subgroup",
    )

The plain read produces the ``A`` fragment, and the ``transform`` read packs
K-adjacent rows into the VNNI words ``B`` needs, in hardware. Each read and write
variant was checked on an Arc Pro B60 and lowering accepts no others; the
:doc:`reference </reference/python/backend.intel.tile.block_io>` lists them.

Two surface parameters silently return wrong data when violated, so lowering
must prove both: the matrix's row pitch is a multiple of 16 bytes, and each
block's x offset is 4-byte aligned. Reads outside the matrix return zero and
writes outside it are dropped, so edge tiles need no masking.

On an Arc Pro B60, this kernel with ``4 x 4`` sub-groups per work-group computes
a ``2048``\ :sup:`3` float16 product at 63--67 TFLOP/s, against 8.6 TFLOP/s
with scalar fragment loads and 2.0 TFLOP/s for a scalar-FMA kernel. These are
medians over interleaved runs; a single run on a shared machine can read far
lower while the clocks ramp up.

B Stored Transposed
-------------------

Model weights are usually stored ``(N, K)``, so a layer computes ``A @ B^T``.
A 32-bit word of a ``B`` row packs two K-adjacent halves, which is exactly a
VNNI word, so declaring ``B`` as ``uint32`` words and reading it with the
transposed block read hands work-item ``n`` row ``n`` as its ``B`` fragment,
with no repacking:

.. code-block:: python

    # Bw: (batch, N, K // 2) uint32, the (N, K) float16 weights two halves per word
    T.opencl.tile.block_read(
        "intel_sub_group_2d_block_read_transpose_32b_16r8x1c",
        br.view(2, 1, 8, 16, layout=fragment(2, 1, 8)),
        Bw[bz, col0 : col0 + 32, kb * 8 : kb * 8 + 8],
        scope="subgroup",
    )

The region covers 32 rows of ``B`` (the output columns) and 8 words (16 values
of K), two 16 x 8 blocks. The leading ``bz`` selects one matrix of a batch;
leading axes of extent 1 are accepted for every block instruction, as long as
the selected matrix starts 4-byte aligned. The register view's repeats are
``(N blocks, K blocks)``, which is the transpose of DPAS's ``(Kr, Nr)``, so the
same registers are passed to ``dpas`` as ``br.view(1, 2, 8, 16, ...)``.

Epilogues
---------

``T.opencl.tile.ewise(function, dst, *inputs, scope="subgroup")`` applies a
scalar ``@T.function`` at every coordinate of a register fragment, so a bias, a
scale or an activation runs on the accumulator before the block write stores it:

.. code-block:: python

    @T.function(private=True)
    def scale_bias_relu(acc: T.float32, bias: T.float32, alpha: T.float32) -> T.float32:
        return T.max(acc * alpha + bias, T.float32(0))

    # bias[j] holds this work-item's column of output block j; stride 0 on the row
    # axes presents it as the whole tile.
    for j in range(2):
        bias[j] = Bias[col0 + j * 16 + lane]
    T.opencl.tile.ewise(
        scale_bias_relu,
        dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
        dr.view(4, 2, 8, 16, layout=fragment(4, 2, 8)),
        bias.view(4, 2, 8, 16, layout=TileLayout(S[(4, 2, 8, 16) : (0, 1, 0, 1 @ sglid)])),
        T.float32(0.5),
        scope="subgroup",
    )

Each work-item updates only the elements it holds, with no data moving between
work-items. Lowering checks, coordinate by coordinate, that every input holds
each coordinate on the same work-item as ``dst`` and at a register that does not
depend on which work-item it is, and rejects a call that would need data from
another work-item. Because nothing is exchanged, the call is safe under control
flow that differs within a sub-group, but only at ``scope="thread"``: at
``"subgroup"`` it is held to the collectives' convergence proof.

Larger Tiles
------------

A sub-group that computes more of ``C`` loads each fragment once for more
matrix instructions. Three changes take the kernel above to a ``64 x 32``
sub-group tile with a K step of 32:

.. code-block:: python

    def lay(shape, strides):
        return TileLayout(S[shape : (*strides, 1 @ sglid)])

    a_read, a_dpas = lay((2, 2, 32, 16), (32, 64, 1)), lay((8, 2, 8, 16), (8, 64, 1))
    b_read, b_dpas = lay((1, 2, 16, 16), (32, 16, 1)), lay((2, 2, 8, 16), (8, 16, 1))
    d = lay((8, 2, 8, 16), (16, 8, 1))
    prefetch = "intel_sub_group_2d_block_prefetch_16b_32r16x2c"

    for kb in T.serial(K // 32, unroll=False):
        k = kb * 32
        T.opencl.tile.block_prefetch(prefetch, A[row0 : row0 + 64, k + 32 : k + 64], scope="subgroup")
        T.opencl.tile.block_prefetch(prefetch, B[k + 32 : k + 64, col0 : col0 + 32], scope="subgroup")
        T.opencl.tile.block_read(
            "intel_sub_group_2d_block_read_16b_32r16x1c",
            ar.view(2, 2, 32, 16, layout=a_read), A[row0 : row0 + 64, k : k + 32],
            scope="subgroup",
        )
        T.opencl.tile.block_read(
            "intel_sub_group_2d_block_read_transform_16b_32r16x1c",
            br.view(T.uint32.dtype).view(1, 2, 16, 16, layout=b_read),
            B[k : k + 32, col0 : col0 + 32], scope="subgroup",
        )
        T.opencl.tile.dpas(
            "intel_sub_group_f16_f16_matrix_mad_k16",
            dr.view(8, 2, 8, 16, layout=d),
            ar.view(8, 2, 8, 16, layout=a_dpas),
            br.view(T.uint32.dtype).view(2, 2, 8, 16, layout=b_dpas),
            dr.view(8, 2, 8, 16, layout=d),
            scope="subgroup",
        )

**One buffer, two views.** A 32-row block read fills four ``A`` fragments at
once, so each register buffer is described twice: as block-read atoms and as
DPAS fragments. Both views are checked, so their layouts must agree. Ordering
``A`` by K block, then 32-row block, then row (offset ``64·kb + 32·rb + r``)
gives each a single stride per axis: rows of consecutive fragments then sit
eight apart in both.

**Keep the K loop rolled.** IGC fully unrolls a loop whose trip count is
constant, and a body of 32 matrix instructions does not survive that:
unrolled, this kernel ran at 48 TFLOP/s, against 84 kept rolled.
``T.serial(n, unroll=False)`` emits ``#pragma unroll 1``.

**Prefetch one step ahead.** ``block_prefetch`` pulls the next K step's blocks
into cache. Prefetching one step ahead gained 25%; two steps ahead lost about
as much, so the distance is worth measuring rather than guessing.

The tile needs more than the 128 registers an Xe work-item has by default. The
``intel/*`` targets build each kernel with ``-cl-intel-enable-auto-large-GRF-mode``,
which gives a kernel 256 registers only when it would otherwise spill; forcing
256 everywhere instead halves occupancy and cost the smaller kernel above a
third of its speed. The option reaches the compiler through the ``opencl``
target's ``build_options``.

On the B60 the ``64 x 32`` kernel computes ``4096``\ :sup:`3` at 88 TFLOP/s and
``2048``\ :sup:`3` at 71 TFLOP/s, against 67 and 63 TFLOP/s for the ``32 x 32``
kernel and 1.5 and 2.0 TFLOP/s for scalar FMA. Intel's oneDNN, through
``torch.matmul`` on the same machine, reaches 93 and 80 TFLOP/s, so the kernel
runs at 95% and 89% of it.

Two directions do not close that gap. A ``64 x 64`` tile cannot fit: its float32
accumulator alone is 256 values per work-item, the entire large register file.
Staging tiles through shared local memory mainly saves global traffic that
sub-groups of a work-group repeat, and removing that duplication outright, by
splitting the prefetch so each tile is fetched once per work-group, gained
nothing measurable: the shared tiles already hit in cache. oneDNN generates GPU
instructions directly, while this kernel goes through OpenCL C and the driver's
compiler, which is where the remaining difference most likely lies.
