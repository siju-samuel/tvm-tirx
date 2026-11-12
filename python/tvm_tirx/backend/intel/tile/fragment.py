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
"""Sub-group register fragments shared by the Intel tile instructions.

Every Intel tile instruction moves or consumes the same kind of operand: a block
of words spread across the 16 work-items of a sub-group, where atom coordinate
``(row, col)`` lives in work-item ``col`` at register component ``row``.  The
DPAS builtins read their operands that way and the 2D block reads produce them
that way, which was established for each builtin on an Arc Pro B60.
:class:`Fragment` checks a caller's register layout against that mapping.
"""

import tvm_ffi

from tvm_tirx.ir import Bind, Range, TensorRegion, TensorStore, Var
from tvm_tirx.ir.prim import Or, Ramp
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer, ConstraintContext, RangeConstraint

SUB_GROUP_SIZE = 16


def context_analyzer(sctx):
    """An analyzer that knows what the dispatch site knows, and a resolver for its variables.

    TileDispatch records each enclosing loop variable with its range and each
    let-bound variable as the one-point range of its value.  Proofs about an
    operand -- that a coordinate is aligned, say -- usually depend on those
    definitions, so ``resolve`` substitutes the bound values (transitively) before
    a condition is checked, and the loop ranges become constraints.  Emitted code
    keeps the original variables; only the proofs look through them.
    """
    values, constraints = {}, ConstraintContext()
    for var, bound in sctx.var_range_map.items():
        extent = bound.extent
        if hasattr(extent, "value") and int(extent.value) == 1:
            values[var] = bound.begin
        else:
            constraints.append(RangeConstraint(var, bound))
    analyzer = Analyzer()
    analyzer.sync_constraint_context(constraints)

    def resolve(expr):
        for _ in range(len(values) + 1):
            updated = tvm_ffi.structural_map(
                expr, (Var, lambda var: values.get(var, var)), order="post"
            )
            if updated.same_as(expr):
                return expr
            expr = updated
        return expr

    return analyzer, constraints, resolve


def vector_type(dtype, lanes):
    return dtype if lanes == 1 else f"{dtype}x{lanes}"


class Fragment:
    """One register operand, checked against the atom's work-item mapping."""

    def __init__(self, name, region, atom_rows, analyzer, layout_analyzer):
        if not isinstance(region, TensorRegion):
            raise TypeError(f"{name}: expected a register tensor region")
        self.name, self.region = name, region
        self.tensor, self.ty = region.source, region.source.ty
        self.dtype = str(self.ty.dtype.dtype)
        self.analyzer, self.layout_analyzer = analyzer, layout_analyzer
        if self.ty.scope != MemoryScope("local"):
            raise ValueError(f"{name}: fragments must be in registers, got {self.ty.scope}")
        self.layout = self.ty.effective_layout()
        if self.layout is None:
            raise ValueError(f"{name}: operand requires a physical layout")
        self.shape = tuple(int(analyzer.simplify(r.extent)) for r in region.region)
        if len(self.shape) < 4 or self.shape[-2:] != (atom_rows, SUB_GROUP_SIZE):
            raise ValueError(
                f"{name}: expected (*batch, R0, R1, {atom_rows}, {SUB_GROUP_SIZE}), "
                f"got {self.shape}"
            )
        for r, size in zip(region.region, self.ty.shape):
            self._require(r.begin >= 0, "region start is out of bounds")
            self._require(r.begin + r.extent <= size, "region exceeds the tensor shape")
        self.rows = atom_rows
        self.raws = {}

    @property
    def repeats(self):
        return self.shape[:-2]

    def require_bits(self, bits, what):
        dtype = self.ty.dtype.dtype
        if dtype.bits != bits or dtype.lanes != 1:
            raise ValueError(
                f"{self.name}: expected scalar {bits}-bit {what}, got {self.dtype}; "
                "pack elements with an explicit dtype view"
            )

    def _require(self, condition, message):
        if not self.analyzer.can_prove(condition):
            raise ValueError(f"{self.name}: {message}: cannot prove {condition}")

    def _physical(self, coords):
        indices = [self.analyzer.simplify(r.begin + i) for r, i in zip(self.region.region, coords)]
        result = self.layout.apply(*indices, shape=self.ty.shape, analyzer=self.layout_analyzer)
        return {key: self.analyzer.simplify(value) for key, value in dict(result).items()}

    def base(self, repeat):
        """Check one repeat's atom against the hardware mapping; return its storage base.

        Work-item ``lane`` and component ``row`` are symbolic over the whole atom,
        so the proof covers every position at once rather than sampling them.
        """
        lane, row = Var("dpas_lane", "int32"), Var("dpas_row", "int32")
        constraints = ConstraintContext()
        constraints.append(RangeConstraint(lane, Range(0, SUB_GROUP_SIZE)))
        constraints.append(RangeConstraint(row, Range(0, self.rows)))
        self.analyzer.sync_constraint_context(constraints)
        try:
            physical = self._physical((*repeat, row, lane))
            for axis in set(physical) - {"m", "sglid"}:
                self._require(
                    physical[axis] == 0, f"atom places values on unexpected axis {axis!r}"
                )
            self._require(
                physical.get("sglid", 0) == lane,
                "atom column must be the work-item index within the sub-group (sglid)",
            )
            origin = self._physical((*repeat, 0, 0)).get("m", 0)
            self._require(
                physical.get("m", 0) == origin + row,
                "each work-item's atom must occupy consecutive registers in row order",
            )
        finally:
            self.analyzer.sync_constraint_context(ConstraintContext())
        return origin

    def storage(self, bindings, dtype=None):
        """A flat alias of the operand's registers, addressed by storage offset.

        ``dtype`` reinterprets the words, which must keep their width; builtins
        that take a pointer to unsigned words need an alias of exactly that type.
        """
        dtype = dtype or self.dtype
        if dtype not in self.raws:
            span = self.layout.span("m")
            raw = Var(
                f"frag_{self.name}" + ("" if dtype == self.dtype else f"_{dtype}"),
                T.Tensor((span,), dtype, scope="local", layout="default"),
            )
            bindings.append(Bind(raw, T.decl_tensor(self.tensor.data, ty_args=[raw.ty])))
            self.raws[dtype] = raw
        return self.raws[dtype]

    def address(self, bindings, base, dtype):
        """A pointer to this fragment's registers at ``base``, as ``dtype`` words."""
        return T.address_of(self.storage(bindings, dtype)[base])

    def load(self, bindings, base, as_dtype):
        raw = self.storage(bindings)
        value = raw[base] if self.rows == 1 else raw[Ramp(base, 1, self.rows)]
        target = vector_type(as_dtype, self.rows)
        return value if self.dtype == as_dtype else T.reinterpret(value, ty=target)

    def store(self, bindings, base, value):
        raw = self.storage(bindings)
        index = base if self.rows == 1 else Ramp(base, 1, self.rows)
        return TensorStore(raw, [index], value)


def require_intel_target(target, what):
    keys = {str(key) for key in target.keys}
    if "intel" not in keys:
        raise ValueError(
            f"{what} requires an Intel OpenCL target, such as intel/arc-b60-opencl; "
            f"got keys {sorted(keys)}"
        )
    warp = int(target.attrs["thread_warp_size"])
    if warp != SUB_GROUP_SIZE:
        raise ValueError(
            f"{what} requires a sub-group size of {SUB_GROUP_SIZE}, "
            f"but the target declares thread_warp_size={warp}"
        )


def check_disjoint(analyzer, bases, width, name):
    """Distinct repeats must write distinct registers, or one result overwrites another."""
    for i, first in enumerate(bases):
        for second in bases[i + 1 :]:
            apart = Or(first - second >= width, second - first >= width)
            if not analyzer.can_prove(apart):
                raise ValueError(f"{name}: two repeats share registers")
