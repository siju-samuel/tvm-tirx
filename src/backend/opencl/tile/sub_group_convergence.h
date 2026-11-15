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
 * \file sub_group_convergence.h
 * \brief Proof that every work-item of a sub-group reaches each sub-group tile instruction.
 */
#ifndef TVM_BACKEND_OPENCL_TILE_SUB_GROUP_CONVERGENCE_H_
#define TVM_BACKEND_OPENCL_TILE_SUB_GROUP_CONVERGENCE_H_

#include <tvm/ir/stmt.h>
#include <tvm/target/target.h>

namespace tvm {
namespace backend {
namespace opencl {

/*!
 * \brief Reject a sub-group tile instruction that some work-items of a sub-group may skip.
 *
 * An instruction at ``opencl.subgroup`` scope is a sub-group collective, and OpenCL leaves a
 * collective undefined unless every work-item of the sub-group reaches it.  CUDA's tile layer
 * proves the matching property for warp instructions; this is the OpenCL counterpart.
 *
 * For each such instruction it proves that the sub-groups are full -- the work-group's x extent is
 * a multiple of the sub-group size -- and that nothing on the way to the instruction can treat
 * work-items of one sub-group differently: enclosing if and while conditions (either branch),
 * enclosing loop bounds, a break or continue in an enclosing loop, or an earlier return.
 *
 * A condition is proved uniform rather than recognized.  ``threadIdx.x`` is rewritten as
 * ``size * q + l`` and the condition must be provably equal for every ``l`` in ``[0, size)``, so
 * ``tx / 16`` passes and ``tx % 16`` does not.  Anything the prover cannot settle counts as
 * divergent: the check may reject a correct kernel, but it does not accept a divergent one.
 *
 * Sub-groups are assumed to be formed from consecutive work-items along x, as Intel's
 * implementation does; OpenCL leaves the grouping to the implementation, and the only sub-group
 * tile instructions so far require an Intel target.
 *
 * \param body The function body, before tile dispatch.
 * \param target The function's target; its thread_warp_size is the sub-group size.
 * \throws ffi::Error (ValueError) naming the instruction and why convergence cannot be proved.
 */
void CheckSubGroupConvergence(const Stmt& body, const Target& target);

}  // namespace opencl
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_OPENCL_TILE_SUB_GROUP_CONVERGENCE_H_
