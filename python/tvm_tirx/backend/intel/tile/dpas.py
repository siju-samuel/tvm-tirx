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
"""Intel Xe matrix multiply-accumulate (DPAS) over sub-group register fragments.

``T.opencl.tile.dpas(instruction, D, A, B, C, scope="subgroup")`` computes
``D = A @ B + C`` with the named ``cl_intel_subgroup_matrix_multiply_accumulate``
builtin.  One sub-group of 16 work-items executes each instruction on fragments
held in its registers.

Operand coordinates are repeat axes followed by the instruction's atom::

    D, C: (*batch, Mr, Nr, M, 16)   32-bit accumulator elements
    A:    (*batch, Mr, Kr, M, 16)   16-bit words
    B:    (*batch, Kr, Nr, 8, 16)   32-bit words

``M`` is 1, 2, 4 or 8 and selects the builtin's overload.  ``Mr``, ``Nr`` and
``Kr`` repeat the instruction; each ``(Mr, Nr)`` position accumulates through
``Kr``, reading ``C`` on the first issue and the preceding ``D`` on the rest.
``C`` may also be a scalar, broadcast across the accumulator.

As in the CUDA tile layer, the dtype describes the storage word and the
instruction does not pack values.  A word of ``A`` is one 16-bit element, or a
pair of 8-bit elements adjacent in K.  A word of ``B`` packs the K-adjacent
elements that fill 32 bits -- two 16-bit or four 8-bit -- so a ``float16`` or
``int8`` B fragment is passed as a ``uint32`` view.  Because packed words do not
carry their element type, the instruction is named explicitly.

Every atom maps its coordinate ``(row, col)`` to the same place: work-item
``col`` of the sub-group, at register component ``row``.  Lowering checks that
each operand's layout puts every atom coordinate on ``sglid == col`` and its
components at consecutive storage offsets, so a fragment that does not match the
hardware fails to compile rather than producing wrong results.  The contract was
established by running each builtin on an Arc Pro B60 against a reference
product.

Like every sub-group collective, the instruction requires all 16 work-items of
the sub-group to reach it.  OpenCL has no execution analysis in TIRx yet, so
unlike CUDA's warp instructions this is not checked: the call must be in
sub-group-uniform control flow, in a work-group whose x extent is a multiple of
16.  The same contract already applies to the sub-group shuffles that thread
reductions lower to on this target.
"""

from itertools import product

from tvm_tirx.ir import Bind, Range, SeqStmt, TensorRegion, TensorStore, Var
from tvm_tirx.ir.prim import Broadcast, Cast, Or, Ramp
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer, ConstraintContext, RangeConstraint
from tvm_tirx.tirx.tile.instruction import Instruction, Operand

SUB_GROUP_SIZE = 16
M_CHOICES = (1, 2, 4, 8)
B_WORDS = 8  # 32-bit words of B per work-item: K * bits(element) / 32.

# Each builtin's operand types at sub-group size 16: the A word, the B word, and
# the accumulator element types it accepts.  OpenCL vectors do not convert
# implicitly, so A and B are reinterpreted to exactly these types; the unsigned
# families take unsigned vectors.  Taken from the overloads the Arc Pro B60
# driver declares.  The ``int``-typed A overloads it also declares are the
# sub-group-8 forms and do not apply here.
FAMILIES = {
    "intel_sub_group_f16_f16_matrix_mad_k16": ("int16", "int32", ("float32", "float16")),
    "intel_sub_group_bf16_bf16_matrix_mad_k16": ("int16", "int32", ("float32",)),
    "intel_sub_group_i8_i8_matrix_mad_k32": ("int16", "int32", ("int32",)),
    "intel_sub_group_u8_u8_matrix_mad_k32": ("uint16", "uint32", ("int32",)),
    "intel_sub_group_u8_i8_matrix_mad_k32": ("uint16", "int32", ("int32",)),
    "intel_sub_group_i8_u8_matrix_mad_k32": ("int16", "uint32", ("int32",)),
}


def _vector(dtype, lanes):
    return dtype if lanes == 1 else f"{dtype}x{lanes}"


class _Fragment:
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
        self.raw = None

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

    def storage(self, bindings):
        """A flat alias of the operand's registers, addressed by storage offset."""
        if self.raw is None:
            span = self.layout.span("m")
            self.raw = Var(
                "dpas_" + self.name,
                T.Tensor((span,), self.dtype, scope="local", layout="default"),
            )
            bindings.append(Bind(self.raw, T.decl_tensor(self.tensor.data, ty_args=[self.raw.ty])))
        return self.raw

    def load(self, bindings, base, as_dtype):
        raw = self.storage(bindings)
        value = raw[base] if self.rows == 1 else raw[Ramp(base, 1, self.rows)]
        target = _vector(as_dtype, self.rows)
        return value if self.dtype == as_dtype else T.reinterpret(value, ty=target)

    def store(self, bindings, base, value):
        raw = self.storage(bindings)
        index = base if self.rows == 1 else Ramp(base, 1, self.rows)
        return TensorStore(raw, [index], value)


def _require_intel_target(target):
    keys = {str(key) for key in target.keys}
    if "intel" not in keys:
        raise ValueError(
            "dpas requires an Intel OpenCL target, such as intel/arc-b60-opencl; "
            f"got keys {sorted(keys)}"
        )
    warp = int(target.attrs["thread_warp_size"])
    if warp != SUB_GROUP_SIZE:
        raise ValueError(
            f"dpas requires a sub-group size of {SUB_GROUP_SIZE}, "
            f"but the target declares thread_warp_size={warp}"
        )


def _check_disjoint(analyzer, bases, width, name):
    """Distinct repeats must write distinct registers, or one result overwrites another."""
    for i, first in enumerate(bases):
        for second in bases[i + 1 :]:
            apart = Or(first - second >= width, second - first >= width)
            if not analyzer.can_prove(apart):
                raise ValueError(f"{name}: two repeats of the accumulator share registers")


def lower(view, sctx):
    """Check the operands of one ``dpas`` call and emit its builtin calls."""
    instruction, d, a, b, c = view.args
    name = instruction.value if hasattr(instruction, "value") else str(instruction)
    if name not in FAMILIES:
        raise ValueError(f"Unknown dpas instruction {name!r}; expected one of {sorted(FAMILIES)}")
    _require_intel_target(sctx.target)
    a_word, b_word, accumulators = FAMILIES[name]

    analyzer, layout_analyzer = Analyzer(), Analyzer()
    if not isinstance(d, TensorRegion) or len(d.region) < 4:
        raise TypeError("D: expected a register tensor region of rank 4 or more")
    m = int(analyzer.simplify(d.region[-2].extent))
    if m not in M_CHOICES:
        raise ValueError(f"D: atom row count must be one of {M_CHOICES}, got {m}")

    df = _Fragment("D", d, m, analyzer, layout_analyzer)
    af = _Fragment("A", a, m, analyzer, layout_analyzer)
    bf = _Fragment("B", b, B_WORDS, analyzer, layout_analyzer)
    cf = _Fragment("C", c, m, analyzer, layout_analyzer) if isinstance(c, TensorRegion) else None

    accumulator = df.dtype
    if accumulator not in accumulators:
        raise ValueError(f"D: {name} accumulates in {' or '.join(accumulators)}, not {accumulator}")
    if cf is not None and cf.dtype != accumulator:
        raise ValueError(f"C: expected the accumulator type {accumulator}, got {cf.dtype}")
    af.require_bits(16, "words")
    bf.require_bits(32, "words")
    if cf is None:
        c = Cast(accumulator, c) if str(c.ty.dtype) != accumulator else c

    *batch, mr, nr = df.repeats
    if len(af.repeats) != len(df.repeats) or tuple(af.repeats[:-1]) != (*batch, mr):
        raise ValueError(f"A: expected repeats (*batch, Mr, Kr) = ({batch}, {mr}, Kr)")
    kr = af.repeats[-1]
    if tuple(bf.repeats) != (*batch, kr, nr):
        raise ValueError(f"B: expected repeats (*batch, Kr, Nr) = {(*batch, kr, nr)}")
    if cf is not None and cf.repeats != df.repeats:
        raise ValueError(f"C: expected the accumulator repeats {df.repeats}")

    batches = list(product(*(range(n) for n in batch)))
    d_bases = {
        (*bi, i, j): df.base((*bi, i, j)) for bi in batches for i in range(mr) for j in range(nr)
    }
    _check_disjoint(analyzer, list(d_bases.values()), m, "D")
    c_bases = {key: cf.base(key) for key in d_bases} if cf is not None else None
    a_bases = {
        (*bi, i, k): af.base((*bi, i, k)) for bi in batches for i in range(mr) for k in range(kr)
    }
    b_bases = {
        (*bi, k, j): bf.base((*bi, k, j)) for bi in batches for k in range(kr) for j in range(nr)
    }

    bindings, body = [], []
    acc_ty = _vector(accumulator, m)
    for (*bi, i, j), d_base in d_bases.items():
        for k in range(kr):
            if k > 0:
                acc = df.load(bindings, d_base, accumulator)
            elif cf is not None:
                acc = cf.load(bindings, c_bases[(*bi, i, j)], accumulator)
            else:
                acc = c if m == 1 else Broadcast(c, m)
            product_ = T.call_pure_extern(
                name,
                af.load(bindings, a_bases[(*bi, i, k)], a_word),
                bf.load(bindings, b_bases[(*bi, k, j)], b_word),
                acc,
                ty=acc_ty,
            )
            body.append(df.store(bindings, d_base, product_))
    return SeqStmt([*bindings, *body])


_construct = Instruction(
    "tirx.opencl.tile.dpas",
    "intel_dpas",
    (
        Operand("instruction"),
        Operand("D", "region"),
        Operand("A", "region"),
        Operand("B", "region"),
        Operand("C"),
    ),
    ("subgroup",),
    lower=lower,
).register()


def _publish(factory):
    """Bind the instruction's own constructor where ``T.opencl.tile.dpas`` resolves.

    Registering an instruction also exposes a generic op constructor under its
    name, and that one hands ``scope="subgroup"`` to the scope enum unqualified.
    The instruction's factory qualifies it to ``opencl.subgroup`` and validates
    the operands, so install it in both exposure destinations: the backend module
    takes the raw factory, the script namespace the auto-emitting one its other
    ops use.  Exposure keeps an existing callable, so later republishing does not
    replace it.
    """
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    from tvm_tirx.script.ir_builder.base import auto_emit_void_call

    for module_name, fn in (
        ("tvm_tirx.backend.opencl", factory),
        ("tvm_tirx.backend.opencl.script", auto_emit_void_call(factory)),
    ):
        tile = import_module(module_name).tile
        tile.dpas = fn


_publish(_construct)
