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
"""A scalar function applied to every element of sub-group register fragments.

``T.intel.tile.ewise(function, dst, *inputs, scope="subgroup")`` computes
``dst[c] = function(inputs[0][c], ...)`` at every logical coordinate ``c`` of
``dst``.  It is how an epilogue -- a bias, a scale, an activation -- runs on a
matrix tile while it is still in the registers DPAS left it in, before a block
write stores it.

The operands are register views like the other Intel instructions take: their
layouts may place an axis on ``sglid``, so one logical tile is spread across the
sub-group.  Each work-item updates only the elements it holds, so the function
is applied with no data moving between work-items.  That is valid when every
input holds each coordinate on the same work-item as ``dst`` does, at a register
that does not depend on which work-item it is; lowering checks exactly that,
coordinate by coordinate.  An input broadcast along an axis -- a per-column bias,
say -- is a view with stride 0 on that axis.  A scalar input is broadcast to
every coordinate.

The function is a scalar ``@T.function`` with typed parameters, scalar
bindings, and a scalar return on every path, as for CUDA's ``ewise``; its
parameter and return types must match the operands.  There is no implicit cast.

At ``scope="subgroup"`` the call is held to the same convergence proof as the
sub-group collectives, which is conservative: no work-item reads another's
registers here.  ``scope="thread"`` lifts it, for use under control flow that
may differ within a sub-group.
"""

from itertools import product

import tvm_ffi

from tvm_tirx.ir import (
    Bind,
    If,
    IntImm,
    LambdaExpr,
    PrimType,
    Return,
    SeqStmt,
    TensorRegion,
    TensorStore,
    Tuple,
    Var,
)
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.tirx import Function
from tvm_tirx.tirx.tile.instruction import Instruction, Operand

from ._publish import publish
from .fragment import SUB_GROUP_SIZE, require_intel_target


def _scalar_body(statements, env):
    """Inline a scalar function body; the same staging the CUDA ewise uses."""

    def substitute(expr):
        return tvm_ffi.structural_map(expr, (Var, lambda v: env.get(v, v)), order="pre")

    for statement in statements.seq:
        if isinstance(statement, Bind) and isinstance(statement.var.ty, PrimType):
            env[statement.var] = substitute(statement.value)
        elif isinstance(statement, Return):
            return substitute(statement.value)
        elif isinstance(statement, If) and statement.else_case is not None:
            return T.if_then_else(
                substitute(statement.cond),
                _scalar_body(statement.then_case, dict(env)),
                _scalar_body(statement.else_case, dict(env)),
            )
        else:
            raise ValueError(
                "ewise function requires scalar bindings and a scalar return on every path"
            )
    raise ValueError("ewise scalar function must return a value")


def _as_lambda(function):
    if isinstance(function, LambdaExpr):
        return function
    if not isinstance(function, Function):
        raise TypeError("ewise requires a scalar TIRx Function")
    if not all(isinstance(p.ty, PrimType) for p in function.params):
        raise TypeError("ewise function parameters must be scalars")

    def expression(*values):
        return _scalar_body(function.body, dict(zip(function.params, values)))

    return LambdaExpr([p.ty for p in function.params], expression, ret_type=function.ret_type)


class _Registers:
    """A register operand's placement of each logical coordinate: (work-item, offset)."""

    def __init__(self, name, region, analyzer):
        if not isinstance(region, TensorRegion):
            raise TypeError(f"{name}: expected a register tensor region")
        self.name, self.region = name, region
        self.tensor, self.ty = region.source, region.source.ty
        self.dtype = str(self.ty.dtype.dtype)
        if self.ty.scope != MemoryScope("local"):
            raise ValueError(f"{name}: ewise operands must be in registers, got {self.ty.scope}")
        self.layout = self.ty.effective_layout()
        if self.layout is None:
            raise ValueError(f"{name}: operand requires a physical layout")
        self.analyzer = analyzer
        self.begin = []
        for r in region.region:
            begin = analyzer.simplify(r.begin)
            if not isinstance(begin, IntImm):
                raise ValueError(f"{name}: ewise needs a static region, got start {begin}")
            self.begin.append(int(begin))
        self.shape = tuple(int(analyzer.simplify(r.extent)) for r in region.region)
        self.raw = None

    def place(self, coord):
        indices = [IntImm("int32", b + c) for b, c in zip(self.begin, coord)]
        physical = dict(self.layout.apply(*indices, shape=self.ty.shape))
        for axis in set(physical) - {"m", "sglid"}:
            if int(self.analyzer.simplify(physical[axis])) != 0:
                raise ValueError(f"{self.name}: layout places values on axis {axis!r}")
        lane = int(self.analyzer.simplify(physical.get("sglid", IntImm("int32", 0))))
        return lane, int(self.analyzer.simplify(physical["m"]))

    def storage(self, bindings):
        if self.raw is None:
            span = self.layout.span("m")
            self.raw = Var(
                f"ewise_{self.name}", T.Tensor((span,), self.dtype, scope="local", layout="default")
            )
            bindings.append(Bind(self.raw, T.decl_tensor(self.tensor.data, ty_args=[self.raw.ty])))
        return self.raw


def _lower(view, sctx):
    function, dst, packed = view.args
    inputs = list(packed.fields) if packed is not None else []
    if len(inputs) != len(function.vars):
        raise ValueError("ewise input count must match the scalar function parameters")
    if sctx.scope_kind == "subgroup":
        require_intel_target(sctx.target, "ewise")
    analyzer = Analyzer()
    out = _Registers("dst", dst, analyzer)
    sources = [
        _Registers(f"src{i}", value, analyzer) if isinstance(value, TensorRegion) else value
        for i, value in enumerate(inputs)
    ]
    for i, (param, value) in enumerate(zip(function.vars, sources)):
        dtype = value.dtype if isinstance(value, _Registers) else str(value.ty.dtype)
        if dtype != str(param.ty.dtype):
            raise TypeError(f"ewise input {i}: dtype must match the scalar function parameter")
        if isinstance(value, _Registers) and value.shape != out.shape:
            raise ValueError(
                f"src{i}: expected the destination's shape {out.shape}, got {value.shape}"
            )

    # Every work-item applies the function at each of its own destination
    # registers.  For that to compute dst[c] = f(src[c]) for every coordinate,
    # each input must hold c on the same work-item, and the register it uses for
    # a given destination register must be the same on every work-item.
    reads = {}  # destination offset -> (work-item -> coordinate), and input offsets
    for coord in product(*(range(n) for n in out.shape)):
        lane, offset = out.place(coord)
        slot = reads.setdefault(offset, {"lanes": {}, "inputs": [None] * len(sources)})
        if lane in slot["lanes"]:
            raise ValueError(f"dst: coordinates {slot['lanes'][lane]} and {coord} share a register")
        slot["lanes"][lane] = coord
        for i, source in enumerate(sources):
            if not isinstance(source, _Registers):
                continue
            src_lane, src_offset = source.place(coord)
            if src_lane != lane:
                raise ValueError(
                    f"src{i}: coordinate {coord} is on work-item {src_lane}, but dst holds it on "
                    f"work-item {lane}; ewise does not move data between work-items"
                )
            if slot["inputs"][i] not in (None, src_offset):
                raise ValueError(
                    f"src{i}: the work-items' coordinates at dst register {offset} sit at "
                    "different registers of src, so no one statement computes them all"
                )
            slot["inputs"][i] = src_offset
    lanes = len({lane for slot in reads.values() for lane in slot["lanes"]})
    if sctx.scope_kind == "subgroup" and lanes not in (1, SUB_GROUP_SIZE):
        raise ValueError(f"dst: the tile spans {lanes} work-items, not a whole sub-group")

    bindings, body = [], []
    out_raw = out.storage(bindings)
    for offset in sorted(reads):
        slot = reads[offset]
        if len(slot["lanes"]) != lanes:
            raise ValueError(f"dst: register {offset} is held by only some of the work-items")
        values = [
            source.storage(bindings)[slot["inputs"][i]]
            if isinstance(source, _Registers)
            else source
            for i, source in enumerate(sources)
        ]
        result = function.apply(values)
        if str(result.ty.dtype) != out.dtype:
            raise TypeError("ewise return dtype must match the destination dtype")
        body.append(TensorStore(out_raw, [IntImm("int32", offset)], result))
    return SeqStmt([*bindings, *body])


_construct = Instruction(
    "tirx.intel.tile.ewise",
    "intel_ewise",
    (Operand("function"), Operand("dst", "region"), Operand("inputs", default=None)),
    ("thread", "subgroup"),
    lower=_lower,
    runs_on="opencl",
).register()


def ewise(function, dst, *inputs, scope="subgroup", **kwargs):
    """Apply a scalar TIRx function at every coordinate of a register fragment.

    Parameters
    ----------
    function : tvm_tirx.tirx.Function or tvm_tirx.ir.LambdaExpr
        A typed scalar function with a scalar return on every path.
    dst : TensorRegion
        Destination registers.
    *inputs : TensorRegion or Expr
        Register regions with the destination's shape, each holding every
        coordinate on the same work-item as ``dst``, or scalars to broadcast.
    scope : str, optional
        ``"subgroup"`` (the default) or ``"thread"``.

    Returns
    -------
    call : tvm_tirx.ir.Call
        The function application over every register coordinate.
    """
    if len(inputs) == 1 and isinstance(inputs[0], Tuple | tuple | list):
        values = inputs[0]
        inputs = tuple(values.fields if isinstance(values, Tuple) else values)
    return _construct(_as_lambda(function), dst, inputs, scope=scope, **kwargs)


publish("ewise", ewise)
