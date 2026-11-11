/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
#include <tvm/codegen/intrin_rule.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

namespace tvm {
namespace codegen {
namespace intrin {
using tirx::FLowerIntrinsic;

// There is no warp shuffle instruction in standard OpenCL.  When shuffle is used, assume Intel's
// shuffle extension.
//
// Every shuffle carries {mask, value, lane_or_delta, width, warp_size}.  The mask is unused: an
// Intel sub-group has no activemask equivalent, and LowerThreadAllreduce is told not to build one
// for this target.  The width must equal the sub-group size, because the Intel builtins always
// shuffle across the whole sub-group.
static void CheckShuffleWidth(const CallObj* call) {
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, lane_or_delta, width, warp_size
  sym::Analyzer analyzer;
  TVM_FFI_ICHECK(analyzer->CanProve(call->args[3].as_or_throw<PrimExpr>() ==
                                    call->args[4].as_or_throw<PrimExpr>()))
      << "Intel warp shuffle does not support width != warp_size";
}

// CUDA defines __shfl_sync's source as ``srcLane % width``, so callers are free to pass a lane
// index counted from the start of the thread block rather than the start of the warp.
// LowerThreadAllreduce does exactly that when it broadcasts a sub-warp reduction: the lane it asks
// for is ``reduce_extent * group_index``, which runs past the sub-group for every group after the
// first.  intel_sub_group_shuffle does not wrap an out-of-range index -- it returned 0 on
// Battlemage -- so reduce the index explicitly to keep CUDA's semantics.
static PrimExpr DispatchIntelShuffle(const PrimExpr& e) {
  const CallObj* call = e.as<CallObj>();
  CheckShuffleWidth(call);
  sym::Analyzer analyzer;
  PrimExpr lane = analyzer->Simplify(
      floormod(call->args[2].as_or_throw<PrimExpr>(), call->args[4].as_or_throw<PrimExpr>()));
  ffi::Array<Expr> opencl_args{StringImm("intel_sub_group_shuffle"),
                               call->args[1].as_or_throw<PrimExpr>(), lane};
  return Call(e.ty(), tirx::call_pure_extern_op(), opencl_args).as_or_throw<PrimExpr>();
}

// ``intel_sub_group_shuffle_down(current, next, delta)`` concatenates two sub-group-wide vectors
// and reads ``delta`` lanes up from each lane's position.  CUDA's ``__shfl_down_sync`` instead
// leaves lanes whose source index would run past the warp holding their own value.  Passing the
// same value as both operands makes the tail lanes read a rotated value rather than their own, so
// the two builtins disagree for lanes near the top of the sub-group.
//
// That difference is immaterial here.  The only consumer is MakeWarpAllreduce, whose tree reduction
// is read back from lane 0 of each reducing group and then broadcast; the tail lanes accumulate
// values nobody loads.  This is the same assumption the CUDA path already relies on.  Any future
// caller that reads a lane other than the group's first must not use this lowering.
//
// Note the third argument is a shift distance, not a lane index, so it must not be reduced modulo
// the sub-group size the way DispatchIntelShuffle's lane index is.
static PrimExpr DispatchIntelShuffleDown(const PrimExpr& e) {
  const CallObj* call = e.as<CallObj>();
  CheckShuffleWidth(call);
  PrimExpr value = call->args[1].as_or_throw<PrimExpr>();
  ffi::Array<Expr> opencl_args{StringImm("intel_sub_group_shuffle_down"), value, value,
                               call->args[2].as_or_throw<PrimExpr>()};
  return Call(e.ty(), tirx::call_pure_extern_op(), opencl_args).as_or_throw<PrimExpr>();
}

// ``intel_sub_group_shuffle_up(previous, current, delta)``, mirroring the above.
static PrimExpr DispatchIntelShuffleUp(const PrimExpr& e) {
  const CallObj* call = e.as<CallObj>();
  CheckShuffleWidth(call);
  PrimExpr value = call->args[1].as_or_throw<PrimExpr>();
  ffi::Array<Expr> opencl_args{StringImm("intel_sub_group_shuffle_up"), value, value,
                               call->args[2].as_or_throw<PrimExpr>()};
  return Call(e.ty(), tirx::call_pure_extern_op(), opencl_args).as_or_throw<PrimExpr>();
}

void RegisterOpenCLIntrinRules() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  // clang-format off
  OpDef("prim.clz")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.fabs")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
        // OpenCL's rint() uses ties-to-even, matching constant-folding semantics.
        const CallObj* call = e.as<CallObj>();
        TVM_FFI_ICHECK(call != nullptr);
        ffi::Array<Expr> new_args = {StringImm("rint")};
        for (const PrimExpr& arg : call->args.as_or_throw<ffi::Array<PrimExpr>>()) {
          new_args.push_back(arg);
        }
        return Call(e.ty(), tirx::call_pure_extern_op(), new_args).as_or_throw<PrimExpr>();
      });

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.erf")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.exp2")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.exp10")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.log10")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.tanh")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.popcount")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.fmod")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.sinh")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.cosh")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.gpu_warp_shuffle")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchIntelShuffle);

  OpDef("tirx.gpu_warp_shuffle_up")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchIntelShuffleUp);

  OpDef("tirx.gpu_warp_shuffle_down")
      .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchIntelShuffleDown);
  // clang-format on
}

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
