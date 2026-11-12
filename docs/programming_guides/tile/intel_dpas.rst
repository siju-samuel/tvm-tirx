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
     - float32
   * - ``intel_sub_group_i8_i8_matrix_mad_k32``, ``u8_u8``, ``u8_i8``, ``i8_u8``
     - 32
     - int32

The unsigned families take unsigned vectors, and the lowering reinterprets each
operand to the exact types the builtin declares.

What Is Checked
---------------

Lowering proves, for each repeat and over every work-item and component of the
atom, that the operand's layout places the value on ``sglid == col`` and that a
work-item's components sit in consecutive registers. It also checks that output
repeats do not share registers, that ``A`` and ``B`` use 16- and 32-bit words,
that the accumulator type suits the builtin, and that the target is an Intel
part with a declared sub-group size of 16. A fragment that disagrees with the
hardware fails to compile with the reason, instead of returning wrong numbers.

The check covers the instruction, not the code that fills the registers. The
loads in the example index by ``lane``, and the lowering takes on trust that the
loaded values match the layout. Like every sub-group collective, the instruction
also requires all 16 work-items of the sub-group to reach it. TIRx has no
OpenCL execution analysis yet, so this is not checked either: call it in
sub-group-uniform control flow, in a work-group whose x extent is a multiple
of 16.

The example loads fragments with scalar loads, which is enough to check
correctness but leaves the XMX engines underfed. The 2D block reads in
``cl_intel_subgroup_2d_block_io`` load a whole fragment per instruction and are
the next step for performance.
