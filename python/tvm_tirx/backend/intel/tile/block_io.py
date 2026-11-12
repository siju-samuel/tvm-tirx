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
"""Intel Xe 2D block reads and writes between global memory and sub-group fragments.

``T.opencl.tile.block_read(instruction, dst, src, scope="subgroup")`` loads a
region of a row-major global matrix into sub-group registers with the named
``cl_intel_subgroup_2d_block_io`` builtin, and ``block_write`` stores registers
back.  One instruction moves a whole block, which is what keeps the XMX engines
fed: a block read replaces the per-element loads a fragment otherwise needs.

The global operand is a 2D region of a row-major tensor.  It is tiled into the
instruction's ``rows x cols`` blocks, and the register operand has one
``(words, 16)`` atom per block, with repeat axes ``(Rr, Cr)`` over the row and
column blocks::

    registers: (Rr, Cr, words, 16)        global: (Rr * rows, Cr * cols)

Every atom uses the mapping the DPAS operands use -- word ``(row, col)`` lives in
work-item ``col`` at register component ``row`` -- so a fragment flows from
``block_read`` to ``T.opencl.tile.dpas`` to ``block_write`` with the same view.
What a word holds depends on the variant:

========================================  ======  ======================================
Builtin                                   Word    Word ``(q, i)`` of a block at (y, x)
========================================  ======  ======================================
``read_16b_{1,2,4,8,16,32}r16x1c``        16-bit  ``M[y+q][x+i]``
``read_8b_{1,2,4,8,16,32}r32x1c``         16-bit  ``M[y+q][x+2i]``, ``M[y+q][x+2i+1]``
``read_transform_16b_{16,32}r16x1c``      32-bit  ``M[y+2q][x+i]``, ``M[y+2q+1][x+i]``
``read_transform_8b_32r16x1c``            32-bit  ``M[y+4q+t][x+i]`` for ``t`` in 0..3
``write_32b_{1,2,4,8}r16x1c``             32-bit  ``M[y+q][x+i]``
``write_16b_{1,2,4,8}r16x1c``             16-bit  ``M[y+q][x+i]``
========================================  ======  ======================================

Packed words list their elements from the low bits.  The plain reads produce the
DPAS ``A`` fragment and the transforms the VNNI-packed ``B`` fragment, in
hardware.  Each variant was checked on an Arc Pro B60 against the reference
block; no others are accepted.

Reads outside the surface return zero and writes outside it are dropped, both
checked on hardware, so a block may overhang the edge of the matrix and edge
tiles need no bounds code.  Two surface parameters silently produce wrong data
when violated, so lowering requires a proof of both: the row pitch must be a
multiple of 16 bytes, and every block's x offset must be 4-byte aligned.  The
tensor must be at least 4-byte aligned, which was the coarsest alignment tested.

As with the DPAS instruction, all 16 work-items of the sub-group must reach the
call, which TIRx does not yet check on OpenCL.
"""

import re
from dataclasses import dataclass

from tvm_tirx.ir import Evaluate, IntImm, SeqStmt, TensorRegion
from tvm_tirx.ir.prim import Cast, Shuffle
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.tirx.tile.instruction import Instruction, Operand

from ._publish import publish
from .fragment import Fragment, check_disjoint, context_analyzer, require_intel_target

_NAME = re.compile(
    r"^intel_sub_group_2d_block_(read|read_transform|write)_(8|16|32)b_(\d+)r(\d+)x1c$"
)

# (kind, element bits) -> (rows the hardware was checked with, block columns).
VERIFIED = {
    ("read", 16): ((1, 2, 4, 8, 16, 32), 16),
    ("read", 8): ((1, 2, 4, 8, 16, 32), 32),
    ("read_transform", 16): ((16, 32), 16),
    ("read_transform", 8): ((32,), 16),
    ("write", 32): ((1, 2, 4, 8), 16),
    ("write", 16): ((1, 2, 4, 8), 16),
}
PITCH_ALIGNMENT = 16  # bytes; anything coarser than 4 bytes but not 16 reads wrong data
X_ALIGNMENT = 4  # bytes
BASE_ALIGNMENT = 4  # bytes; the coarsest alignment measured to work


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    bits: int  # element bits in global memory
    rows: int  # block rows in global memory
    cols: int  # block columns in global memory, in elements
    words: int  # register words per work-item
    word_bits: int

    @property
    def is_read(self):
        return self.kind != "write"


def variant(name):
    """Parse and validate a block I/O builtin name."""
    match = _NAME.match(name)
    if match is None:
        raise ValueError(f"Unknown 2D block instruction {name!r}")
    kind, bits, rows, cols = match.group(1), *(int(g) for g in match.group(2, 3, 4))
    allowed = VERIFIED.get((kind, bits))
    if allowed is None or rows not in allowed[0] or cols != allowed[1]:
        raise ValueError(f"{name}: this 2D block variant has not been verified")
    if kind == "read_transform":
        word_bits, words = 32, rows * bits // 32
    elif kind == "read":
        word_bits, words = 16 if bits <= 16 else 32, rows
    else:
        word_bits, words = bits, rows
    return Variant(name, kind, bits, rows, cols, words, word_bits)


class _Surface:
    """The global matrix a block instruction addresses, and the region within it."""

    def __init__(self, region, var, analyzer, resolve):
        if not isinstance(region, TensorRegion):
            raise TypeError("global operand: expected a tensor region")
        tensor, ty = region.source, region.source.ty
        if ty.scope != MemoryScope("global"):
            raise ValueError(f"global operand: expected global memory, got {ty.scope}")
        if len(ty.shape) != 2:
            raise ValueError(f"global operand: expected a 2D tensor, got shape {tuple(ty.shape)}")
        dtype = ty.dtype.dtype
        if dtype.bits != var.bits or dtype.lanes != 1:
            raise ValueError(
                f"global operand: {var.name} moves {var.bits}-bit elements, got {dtype}"
            )
        self.analyzer, self.tensor, self.resolve = analyzer, tensor, resolve
        nbytes = var.bits // 8

        layout = ty.effective_layout()
        if layout is None:
            raise ValueError("global operand: requires a physical layout")

        def offset(row, col):
            physical = dict(
                layout.apply(IntImm("int32", row), IntImm("int32", col), shape=ty.shape)
            )
            if set(physical) - {"m"}:
                raise ValueError("global operand: layout must address memory only")
            return analyzer.simplify(physical["m"])

        origin = offset(0, 0)
        row_stride = analyzer.simplify(offset(1, 0) - origin)
        self._require(origin == 0, "tensor must start at its storage base")
        self._require(offset(0, 1) - origin == 1, "rows must be contiguous")
        self.pitch = analyzer.simplify(row_stride * nbytes)
        self._require(
            self.pitch % PITCH_ALIGNMENT == 0,
            f"row pitch must be a multiple of {PITCH_ALIGNMENT} bytes",
        )
        if int(ty.data_alignment) < BASE_ALIGNMENT:
            raise ValueError(
                f"global operand: tensor must be at least {BASE_ALIGNMENT}-byte aligned, "
                f"got {ty.data_alignment}"
            )
        self.width = analyzer.simplify(ty.shape[1] * nbytes)
        self.height = ty.shape[0]

        self.y0, self.x0 = (analyzer.simplify(r.begin) for r in region.region)
        self.extent = tuple(int(analyzer.simplify(r.extent)) for r in region.region)
        if self.extent[0] % var.rows or self.extent[1] % var.cols:
            raise ValueError(
                f"global operand: region {self.extent} is not a whole number of "
                f"{var.rows} x {var.cols} blocks"
            )
        # Each block's x offset steps by cols * nbytes, which is a multiple of 4,
        # so checking the first block covers every block.
        self._require(
            self.x0 * nbytes % X_ALIGNMENT == 0,
            f"block x offset must be {X_ALIGNMENT}-byte aligned",
        )
        self.nbytes = nbytes

    def _require(self, condition, message):
        if not self.analyzer.can_prove(self.resolve(condition)):
            raise ValueError(f"global operand: {message}: cannot prove {condition}")

    def coordinate(self, var, row_block, col_block):
        x = Cast("int32", self.x0 + col_block * var.cols)
        y = Cast("int32", self.y0 + row_block * var.rows)
        return Shuffle([x, y], [IntImm("int32", 0), IntImm("int32", 1)])


def _lower(view, sctx, direction):
    instruction, dst, src = view.args
    name = instruction.value if hasattr(instruction, "value") else str(instruction)
    var = variant(name)
    if var.is_read != (direction == "read"):
        expected = "block_read" if var.is_read else "block_write"
        raise ValueError(f"{name} belongs to {expected}")
    require_intel_target(sctx.target, f"block_{direction}")
    registers, memory = (dst, src) if var.is_read else (src, dst)

    analyzer, _, resolve = context_analyzer(sctx)
    layout_analyzer = Analyzer()
    surface = _Surface(memory, var, analyzer, resolve)
    fragment = Fragment("registers", registers, var.words, analyzer, layout_analyzer)
    fragment.require_bits(var.word_bits, "words")
    blocks = (surface.extent[0] // var.rows, surface.extent[1] // var.cols)
    if fragment.repeats != blocks:
        raise ValueError(
            f"registers: global region {surface.extent} is {blocks} blocks of "
            f"{var.rows} x {var.cols}, so expected repeats {blocks}, got {fragment.repeats}"
        )
    bases = {(i, j): fragment.base((i, j)) for i in range(blocks[0]) for j in range(blocks[1])}
    if var.is_read:
        check_disjoint(analyzer, list(bases.values()), var.words, "registers")

    word = f"uint{var.word_bits}"
    width, height, pitch = (
        Cast("int32", x) for x in (surface.width, surface.height, surface.pitch)
    )
    bindings, body = [], []
    for (i, j), base in bases.items():
        call = T.call_extern(
            name,
            surface.tensor.data,
            width,
            height,
            pitch,
            surface.coordinate(var, i, j),
            fragment.address(bindings, base, word),
            ty="void",
        )
        body.append(Evaluate(call))
    return SeqStmt([*bindings, *body])


def _instruction(direction):
    return Instruction(
        f"tirx.opencl.tile.block_{direction}",
        f"intel_block_{direction}",
        (Operand("instruction"), Operand("dst", "region"), Operand("src", "region")),
        ("subgroup",),
        lower=lambda view, sctx: _lower(view, sctx, direction),
    ).register()


publish("block_read", _instruction("read"))
publish("block_write", _instruction("write"))
