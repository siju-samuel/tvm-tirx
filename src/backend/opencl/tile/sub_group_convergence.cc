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
 * \file sub_group_convergence.cc
 * \brief Proof that every work-item of a sub-group reaches each sub-group tile instruction.
 */
#include "sub_group_convergence.h"

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/constraint.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tile/dispatch.h>
#include <tvm/tirx/tile/thread_scope.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace backend {
namespace opencl {
namespace {

using tirx::StmtExprVisitor;

std::string Show(const ffi::ObjectRef& node) {
  std::ostringstream os;
  os << node;
  return os.str();
}

/*!
 * \brief Whether an expression reads something that can differ between work-items with the same
 * threadIdx.x substitution: a work-item index builtin, an impure extern, or a private-memory load.
 * A private load at a uniform index still reads each work-item's own register, so it is never
 * uniform.
 */
class LaneDependentRead : public StmtExprVisitor {
 public:
  LaneDependentRead() {}

  bool Check(const PrimExpr& expr) {
    found_ = false;
    StmtExprVisitor::Visit(expr);
    return found_;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    static const std::unordered_set<std::string> lane_builtins = {
        "get_sub_group_local_id", "get_local_id",         "get_global_id",
        "get_local_linear_id",    "get_global_linear_id",
    };
    if (op->op.same_as(tirx::call_extern_op())) {
      found_ = true;
    } else if (op->op.same_as(tirx::call_pure_extern_op())) {
      if (const auto* name = op->args[0].as<StringImmObj>();
          name && lane_builtins.count(std::string(name->value))) {
        found_ = true;
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    static const auto local = runtime::MemoryScope::Get("local");
    if (auto tensor = op->source.as<tirx::TensorVar>(); tensor && tensor.value().scope() == local) {
      found_ = true;
    }
    return StmtExprVisitor::Visit_(op);
  }

 private:
  bool found_{false};
};

class ConvergenceChecker : public StmtExprVisitor {
 public:
  explicit ConvergenceChecker(int64_t size) : size_(size) {}

  void Run(const Stmt& body) { StmtExprVisitor::Visit(body); }

  /*! \brief Whether \p expr is provably the same for every work-item of a sub-group. */
  bool Uniform(const PrimExpr& expr) {
    if (!x_var_.has_value()) return true;  // Check() reports the missing x axis itself.
    PrimExpr resolved = Resolve(expr);
    if (ffi::make_object<LaneDependentRead>()->Check(resolved)) return false;
    PrimVar x = x_var_.value();
    PrimVar q("sub_group", x.ty().as_or_throw<PrimType>());
    PrimVar lane("lane", x.ty().as_or_throw<PrimType>());
    PrimExpr size = IntImm(x.ty(), size_);
    auto at = [&](PrimExpr l) {
      return tirx::SubstituteWithDataTypeLegalization(
          resolved, [&](const Var& var) -> ffi::Optional<PrimExpr> {
            if (var.same_as(x)) return q * size + l;
            return std::nullopt;
          });
    };
    PrimExpr differs = prim::cast(at(lane), PrimType::Int(64)) !=
                       prim::cast(at(IntImm(x.ty(), 0)), PrimType::Int(64));
    sym::Analyzer analyzer;
    auto context = analyzer->GetConstraintContext();
    for (const auto& [var, range] : ranges_) {
      if (!var.same_as(x)) context->Append(sym::RangeConstraint(var, range));
    }
    context->Append(sym::RangeConstraint(
        q, Range::FromMinExtent(IntImm(x.ty(), 0), floordiv(x_extent_, size))));
    context->Append(sym::RangeConstraint(lane, Range::FromMinExtent(IntImm(x.ty(), 0), size)));
    analyzer->SyncConstraintContext(context);
    return analyzer->CanProve(!differs);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (!op->op.same_as(tirx::launch_thread_op())) return StmtExprVisitor::Visit_(op);
    PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
    PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
    if (op->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      x_var_ = var;
      x_extent_ = extent;
    }
    ranges_.emplace_back(var, Range::FromMinExtent(IntImm(var.ty(), 0), extent));
    auto result = StmtExprVisitor::Visit_(op);
    ranges_.pop_back();
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (auto value = op->value.as<PrimExpr>())
      values_.insert_or_assign(op->var.get(), value.value());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    Var var = op->var.as_or_throw<Var>();
    size_t mark = reasons_.size();
    std::string loop = "the enclosing loop over `" + Show(var) + "`";
    if (!Uniform(op->begin) || !Uniform(op->end) || !Uniform(op->step)) {
      reasons_.push_back(loop + " has bounds that can differ between work-items of a sub-group");
    } else if (HasDivergentExit(op->body)) {
      reasons_.push_back(loop + " contains a break, continue or return that only some " +
                         "work-items of a sub-group may take");
    }
    ranges_.emplace_back(var, Range(op->begin, op->end));
    auto result = StmtExprVisitor::Visit_(op);
    ranges_.pop_back();
    reasons_.resize(mark);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op) final {
    size_t mark = reasons_.size();
    if (!Uniform(op->condition)) {
      reasons_.push_back("it is inside `while (" + Show(op->condition) +
                         ")`, whose condition can differ between work-items of a sub-group");
    } else if (HasDivergentExit(op->body)) {
      reasons_.push_back("the enclosing while loop contains a break, continue or return that " +
                         std::string("only some work-items of a sub-group may take"));
    }
    auto result = StmtExprVisitor::Visit_(op);
    reasons_.resize(mark);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) final {
    bool uniform = Uniform(op->condition);
    std::string where = "`if (" + Show(op->condition) + ")`";
    std::string varies = ", whose condition can differ between work-items of a sub-group";
    size_t mark = reasons_.size();
    if (!uniform) reasons_.push_back("it is inside " + where + varies);
    if (auto interrupt = StmtExprVisitor::Visit(op->then_case)) return interrupt;
    reasons_.resize(mark);
    if (op->else_case.has_value()) {
      if (!uniform) reasons_.push_back("it is in the else branch of " + where + varies);
      if (auto interrupt = StmtExprVisitor::Visit(op->else_case.value())) return interrupt;
      reasons_.resize(mark);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ReturnObj* op) final {
    // Work-items that return here never reach anything after it.
    if (!reasons_.empty() && exited_.empty()) {
      exited_ = "an earlier return may have been taken by only some work-items of a sub-group (" +
                reasons_.back() + ")";
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj* op) final {
    if (const auto* call = op->value.as<CallObj>(); call && IsSubGroupInstruction(call)) {
      Check(call);
    }
    return StmtExprVisitor::Visit_(op);
  }

 private:
  bool IsSubGroupInstruction(const CallObj* call) const {
    const auto* op = call->op.as<OpObj>();
    if (op == nullptr || !Op::HasAttrMap(tirx::tile::op_attr::kTileOpDispatch)) return false;
    static const auto& dispatchers = Op::GetAttrMap(tirx::tile::op_attr::kTileOpDispatch);
    if (!dispatchers.count(ffi::GetRef<Op>(op)) || !call->attrs.has_value()) return false;
    auto scope =
        ffi::reflection::FieldGetter(call->attrs->GetTypeKey(), "scope")(call->attrs.value())
            .cast<tirx::ThreadScope>();
    return scope->backend() == "opencl" && scope->local_name() == "subgroup";
  }

  void Check(const CallObj* call) {
    std::string reason;
    if (!x_var_.has_value()) {
      reason = "the kernel launches no threadIdx.x axis, so its sub-groups cannot be shown full";
    } else {
      sym::Analyzer analyzer;
      if (!analyzer->CanProve(floormod(x_extent_, IntImm(x_extent_.ty(), size_)) == 0)) {
        reason = "the work-group's x extent, " + Show(x_extent_) + ", is not a multiple of the " +
                 "sub-group size " + std::to_string(size_) + ", so a sub-group can be partial";
      } else if (!exited_.empty()) {
        reason = exited_;
      } else if (!reasons_.empty()) {
        reason = reasons_.back();
      }
    }
    if (!reason.empty()) {
      TVM_FFI_THROW(ValueError) << call->op.as<OpObj>()->name << ": cannot prove that all " << size_
                                << " work-items of each sub-group reach this "
                                << "instruction: " << reason;
    }
  }

  /*! \brief Substitute let-bound values, transitively, so a proof sees through them. */
  PrimExpr Resolve(PrimExpr expr) const {
    for (size_t step = 0; step <= values_.size(); ++step) {
      bool changed = false;
      PrimExpr next = tirx::SubstituteWithDataTypeLegalization(
          expr, [&](const Var& var) -> ffi::Optional<PrimExpr> {
            auto it = values_.find(var.get());
            if (it == values_.end()) return std::nullopt;
            changed = true;
            return it->second;
          });
      if (!changed) return expr;
      expr = next;
    }
    return expr;
  }

  /*!
   * \brief Whether a loop body can let some work-items of a sub-group leave the loop, or the
   * kernel, early.  A break or continue in a nested loop leaves only that loop, which is checked
   * when the nested loop is visited; a return leaves everything.
   */
  bool HasDivergentExit(const Stmt& body);

  int64_t size_;
  ffi::Optional<PrimVar> x_var_;
  PrimExpr x_extent_ = IntImm::Int32(1);
  std::vector<std::pair<Var, Range>> ranges_;
  std::unordered_map<const VarObj*, PrimExpr> values_;
  // Why the current point is not convergent, innermost last; empty when it is.
  std::vector<std::string> reasons_;
  // Set once a return under divergent control flow has been passed.
  std::string exited_;
};

class ExitScanner : public StmtExprVisitor {
 public:
  explicit ExitScanner(ConvergenceChecker* checker) : checker_(checker) {}

  bool Scan(const Stmt& body) {
    StmtExprVisitor::Visit(body);
    return found_;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) final {
    bool divergent = !checker_->Uniform(op->condition);
    divergent_ += divergent;
    auto result = StmtExprVisitor::Visit_(op);
    divergent_ -= divergent;
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    bool divergent = !checker_->Uniform(op->begin) || !checker_->Uniform(op->end);
    divergent_ += divergent;
    ++depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --depth_;
    divergent_ -= divergent;
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op) final {
    bool divergent = !checker_->Uniform(op->condition);
    divergent_ += divergent;
    ++depth_;
    auto result = StmtExprVisitor::Visit_(op);
    --depth_;
    divergent_ -= divergent;
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BreakObj* op) final {
    found_ |= depth_ == 0 && divergent_ > 0;
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ContinueObj* op) final {
    found_ |= depth_ == 0 && divergent_ > 0;
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ReturnObj* op) final {
    found_ |= divergent_ > 0;
    return StmtExprVisitor::Visit_(op);
  }

 private:
  ConvergenceChecker* checker_;
  int divergent_{0};
  int depth_{0};
  bool found_{false};
};

bool ConvergenceChecker::HasDivergentExit(const Stmt& body) {
  return ffi::make_object<ExitScanner>(this)->Scan(body);
}

}  // namespace

void CheckSubGroupConvergence(const Stmt& body, const Target& target) {
  int64_t size = target->GetAttr<int64_t>("thread_warp_size", 1).value();
  // Without a declared sub-group width there is nothing to prove; the sub-group instructions
  // reject such a target themselves.
  if (size <= 1) return;
  ffi::make_object<ConvergenceChecker>(size)->Run(body);
}

}  // namespace opencl
}  // namespace backend
}  // namespace tvm
