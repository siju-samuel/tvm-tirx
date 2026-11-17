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
 * \file timer.cc
 * \brief Starting, stopping and reading the device timers backends register.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/timer.h>

namespace tvm {
namespace runtime {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Timers are only made by the backends; registering the type lets Python hold one.
  refl::ObjectDef<TimerObj>(refl::init(false));
  // Each GPU runtime registers "runtime.timer.<device>", which makes a Timer that measures
  // work on the device itself (CUDA and Vulkan events, OpenCL profiling events), so host
  // launch overhead is left out.  These make such a timer usable from Python.
  refl::GlobalDef()
      .def("runtime.TimerStart", [](Timer timer) { timer->Start(); })
      .def("runtime.TimerStop", [](Timer timer) { timer->Stop(); })
      .def("runtime.TimerSyncAndGetElapsedNanos",
           [](Timer timer) { return timer->SyncAndGetElapsedNanos(); });
}

}  // namespace runtime
}  // namespace tvm
