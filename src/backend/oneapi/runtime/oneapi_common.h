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
 * \file oneapi_common.h
 * \brief Shared state of the oneAPI (SYCL) runtime: the devices, their queues and the current
 *        device.
 *
 * Device ``i`` of type kDLOneAPI is entry ``i`` of SYCL's global device list, which is the
 * index PyTorch reports for an XPU tensor through DLPack.  Memory is unified shared memory
 * allocated in each platform's default context, the context PyTorch's tensors live in, so a
 * tensor PyTorch made and a kernel this runtime launches agree on what a pointer means.
 */
#ifndef TVM_RUNTIME_ONEAPI_ONEAPI_COMMON_H_
#define TVM_RUNTIME_ONEAPI_ONEAPI_COMMON_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace tvm {
namespace runtime {
namespace oneapi {

/*! \brief The process's SYCL devices, with one default in-order queue each. */
class Workspace {
 public:
  static Workspace* Global() {
    static Workspace* inst = new Workspace();
    return inst;
  }

  int num_devices() const { return static_cast<int>(devices_.size()); }

  sycl::device device(int id) const {
    TVM_FFI_CHECK(id >= 0 && id < num_devices(), ValueError)
        << "oneAPI device " << id << " does not exist; SYCL lists " << num_devices();
    return devices_[id];
  }

  sycl::context context(int id) const {
    return device(id).get_platform().khr_get_default_context();
  }

  /*! \brief The queue work for device ``id`` goes to: the current stream, else the default. */
  sycl::queue* queue(int id) {
    if (void* stream = TVMFFIEnvGetStream(kDLOneAPI, id)) {
      return static_cast<sycl::queue*>(stream);
    }
    return default_queue(id);
  }

  sycl::queue* default_queue(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.size() < devices_.size()) queues_.resize(devices_.size());
    if (!queues_[id]) {
      queues_[id] =
          std::make_unique<sycl::queue>(context(id), device(id), sycl::property::queue::in_order{});
    }
    return queues_[id].get();
  }

 private:
  Workspace() : devices_(sycl::device::get_devices()) {}

  std::vector<sycl::device> devices_;
  std::vector<std::unique_ptr<sycl::queue>> queues_;
  std::mutex mutex_;
};

/*! \brief The device the calling thread works on, as SetDevice last set it. */
inline int& CurrentDevice() {
  static thread_local int device_id = 0;
  return device_id;
}

}  // namespace oneapi
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_ONEAPI_ONEAPI_COMMON_H_
