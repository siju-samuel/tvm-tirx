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
"""Reductions over sub-group register fragments: ``T.intel.tile.reduce``.

``T.intel.tile.reduce(op, dst, src, scope="subgroup")`` combines the elements
of ``src`` over every axis on which ``dst`` has extent 1, with ``op`` one of
``"sum"``, ``"max"`` and ``"min"``; the other axes must match.  It is the row
maximum and row sum of a softmax over a DPAS accumulator, whose columns are
spread across the sub-group.

Each work-item first combines the elements it holds.  When the elements reduced
into one result sit on several work-items, a ``sub_group_reduce_<op>`` then
combines across the whole sub-group and leaves the result on every work-item,
so ``dst``'s layout must replicate it there, ``R[16 : 1 @ sglid]``: each
work-item holds every result, ready to be an input of ``ewise``.  When they sit
on one work-item each, no data moves, and ``dst`` keeps ``src``'s placement on
the remaining axes.

Lowering places every coordinate of both operands and checks that one
statement per destination register computes it on every work-item.  A
reduction across work-items needs ``scope="subgroup"``, and with it the
convergence proof the other sub-group collectives get.
"""

from itertools import product

from tvm_tirx.ir import IntImm, SeqStmt, TensorStore
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.tirx.tile.instruction import Instruction, Operand

from ._publish import publish
from .fragment import SUB_GROUP_SIZE, RegisterView, require_converged, require_intel_target

# The combining function, and the OpenCL sub-group builtin that combines across work-items.
_OPS = {
    "sum": (lambda a, b: a + b, "sub_group_reduce_add"),
    "max": (T.max, "sub_group_reduce_max"),
    "min": (T.min, "sub_group_reduce_min"),
}
# Element types the sub-group builtins take.
_DTYPES = ("float32", "float16", "int32", "uint32")


def _lower(view, sctx):
    op, dst, src = view.args
    op = op.value if hasattr(op, "value") else str(op)
    if op not in _OPS:
        raise ValueError(f"reduce: op must be one of {sorted(_OPS)}, got {op!r}")
    require_intel_target(sctx.target, "reduce")
    analyzer = Analyzer()
    out = RegisterView("dst", dst, analyzer)
    inp = RegisterView("src", src, analyzer)
    if out.dtype != inp.dtype:
        raise TypeError(f"reduce: dst is {out.dtype} but src is {inp.dtype}")
    if out.dtype not in _DTYPES:
        raise TypeError(f"reduce: {out.dtype} is not one of {_DTYPES}")
    if len(out.shape) != len(inp.shape) or any(
        o not in (1, i) for o, i in zip(out.shape, inp.shape)
    ):
        raise ValueError(
            f"reduce: dst {out.shape} must match src {inp.shape} except on reduced axes of extent 1"
        )
    reduced = [a for a, (o, i) in enumerate(zip(out.shape, inp.shape)) if o == 1 and i != 1]

    slots = {}  # dst register -> the work-items that hold it, and what it reduces
    for d in product(*(range(n) for n in out.shape)):
        lane, offset = out.place(d)
        ranges = [range(n) if a in reduced else (d[a],) for a, n in enumerate(inp.shape)]
        held = {}  # src work-item -> the registers it holds of the elements reduced into d
        for c in product(*ranges):
            src_lane, src_offset = inp.place(c)
            held.setdefault(src_lane, []).append(src_offset)
        if len(held) == 1:
            ((src_lane, registers),) = held.items()
            across = False
            if src_lane is not None and src_lane != lane:
                where = "every work-item" if lane is None else f"work-item {lane}"
                raise ValueError(
                    f"reduce: the elements of dst {d} are on work-item {src_lane}, but dst holds "
                    f"it on {where}; only a reduction over the sub-group moves data"
                )
        else:
            if len(held) != SUB_GROUP_SIZE or None in held:
                raise ValueError(
                    f"reduce: the elements of dst {d} are on {len(held)} work-items, not the "
                    f"whole sub-group of {SUB_GROUP_SIZE}"
                )
            shapes = {tuple(sorted(r)) for r in held.values()}
            if len(shapes) != 1:
                raise ValueError(
                    f"reduce: the work-items hold the elements of dst {d} at different registers"
                )
            registers = list(shapes.pop())
            across = True
            if lane is not None:
                raise ValueError(
                    f"reduce: dst {d} combines elements from the whole sub-group, which leaves "
                    "the result on every work-item; dst's layout must replicate it there, "
                    "R[16 : 1 @ sglid]"
                )
        plan = (tuple(sorted(registers)), across)
        slot = slots.setdefault(offset, {"lanes": {}, "plan": plan})
        if lane in slot["lanes"]:
            raise ValueError(f"reduce: dst {slot['lanes'][lane]} and {d} share a register")
        slot["lanes"][lane] = d
        if slot["plan"] != plan:
            raise ValueError(
                f"reduce: the work-items' results at dst register {offset} reduce different "
                "registers, so no one statement computes them all"
            )
    lanes = {lane for slot in slots.values() for lane in slot["lanes"]}
    for offset, slot in slots.items():
        if len(slot["lanes"]) != len(lanes):
            raise ValueError(f"reduce: dst register {offset} is held by only some work-items")
    if any(slot["plan"][1] for slot in slots.values()):
        if sctx.scope_kind != "subgroup":
            raise ValueError('reduce: a reduction across work-items needs scope="subgroup"')
        require_converged(sctx, "reduce")

    combine, builtin = _OPS[op]
    bindings = []
    out_raw = out.storage(bindings, "reduce")
    in_raw = inp.storage(bindings, "reduce")
    body = []
    for offset in sorted(slots):
        registers, across = slots[offset]["plan"]
        value = in_raw[IntImm("int32", registers[0])]
        for r in registers[1:]:
            value = combine(value, in_raw[IntImm("int32", r)])
        if across:
            value = T.call_pure_extern(builtin, value, ty=out.dtype)
        body.append(TensorStore(out_raw, [IntImm("int32", offset)], value))
    return SeqStmt([*bindings, *body])


_construct = Instruction(
    "tirx.intel.tile.reduce",
    "intel_reduce",
    (Operand("op"), Operand("dst", "region"), Operand("src", "region")),
    ("thread", "subgroup"),
    lower=_lower,
    runs_on="opencl",
).register()


def reduce(op, dst, src, scope="subgroup", **kwargs):
    """Combine a register fragment over the axes on which ``dst`` has extent 1.

    Parameters
    ----------
    op : {"sum", "max", "min"}
        How elements combine.
    dst : TensorRegion
        Destination registers, with ``src``'s shape except for extent 1 on the
        reduced axes.  A reduction across the sub-group leaves each result on
        every work-item, so then ``dst``'s layout replicates over the
        sub-group, ``R[16 : 1 @ sglid]``.
    src : TensorRegion
        Source registers.
    scope : str, optional
        ``"subgroup"`` (the default) or ``"thread"``, which allows only
        reductions within each work-item's registers.

    Returns
    -------
    call : tvm_tirx.ir.Call
        The reduction.
    """
    return _construct(op, dst, src, scope=scope, **kwargs)


publish("reduce", reduce)
