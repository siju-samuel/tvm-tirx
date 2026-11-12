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
 * \file axis.cc
 * \brief OpenCL layout axes.
 *
 * ``sglid`` is the work-item's index within its sub-group, OpenCL's
 * ``get_sub_group_local_id()``.  A register layout that places a coordinate on
 * ``sglid`` says which work-item of the sub-group holds that value, which is what
 * a sub-group collective such as a matrix multiply-accumulate needs to verify
 * before it reads the operand.
 *
 * The axis is vendor-neutral: sub-groups are core OpenCL.  Instructions that
 * consume it check their own vendor requirements against the target.
 */

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/tirx/layout.h>

#include "../../../tirx/ir/layout/axis_registry.h"

namespace tvm::backend::opencl {

void RegisterLayoutAxes() {
  tvm::tirx::AxisDef("sglid")
      .set_attr("backend", ffi::String("opencl"))
      .set_attr("thread", true)
      .set_attr("scope", tvm::tirx::ThreadScope("opencl.subgroup"))
      .set_attr("subscope", tvm::tirx::ThreadScope("opencl.thread"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tirx.opencl.RegisterLayoutAxes", RegisterLayoutAxes);
}

}  // namespace tvm::backend::opencl
