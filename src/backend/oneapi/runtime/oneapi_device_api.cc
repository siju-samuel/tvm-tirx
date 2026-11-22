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
 * \file oneapi_device_api.cc
 * \brief DeviceAPI for kDLOneAPI devices through SYCL.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory_scope.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

#include "oneapi_common.h"

namespace tvm {
namespace runtime {
namespace oneapi {

class OneAPIDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { CurrentDevice() = dev.device_id; }

  ffi::Any GetAttr(Device dev, const ffi::String& name) final {
    Workspace* w = Workspace::Global();
    if (name == "exist") {
      return dev.device_id >= 0 && dev.device_id < w->num_devices();
    }
    sycl::device d = w->device(dev.device_id);
    namespace info = sycl::info::device;
    if (name == "max_threads_per_block") {
      return static_cast<int64_t>(d.get_info<info::max_work_group_size>());
    }
    if (name == "warp_size") {
      // As for OpenCL: the sub-group size a kernel gets is the compiler's choice, so the
      // runtime does not report one.  The intel/* targets declare theirs.
      return static_cast<int64_t>(1);
    }
    if (name == "max_shared_memory_per_block") {
      return static_cast<int64_t>(d.get_info<info::local_mem_size>());
    }
    if (name == "device_name") {
      return ffi::String(d.get_info<info::name>());
    }
    if (name == "max_clock_rate") {
      return static_cast<int64_t>(d.get_info<info::max_clock_frequency>()) * 1000;
    }
    if (name == "multi_processor_count") {
      return static_cast<int64_t>(d.get_info<info::max_compute_units>());
    }
    if (name == "max_thread_dimensions") {
      auto dims = d.get_info<info::max_work_item_sizes<3>>();
      std::ostringstream os;
      // TVM lists the dimensions x first; SYCL lists them x last.
      os << "[" << dims[2] << ", " << dims[1] << ", " << dims[0] << "]";
      return ffi::String(os.str());
    }
    if (name == "total_global_memory") {
      return static_cast<int64_t>(d.get_info<info::global_mem_size>());
    }
    if (name == "driver_version") {
      return ffi::String(d.get_info<info::driver_version>());
    }
    return nullptr;
  }

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<MemoryScope> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == MemoryScope::Global(), ValueError)
        << "Unsupported memory scope: " << mem_scope.value_or(MemoryScope::Global()).name();
    struct Allocator {
      void AllocData(DLTensor* tensor) {
        tensor->data = Alloc(tensor->device, std::max<size_t>(ffi::GetDataSize(*tensor), 1));
      }
      void FreeData(DLTensor* tensor) { Free(tensor->device, tensor->data); }
    };
    return ffi::Tensor::FromNDAlloc(Allocator{}, shape, dtype, device);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    Workspace* w = Workspace::Global();
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : w->queue(dev.device_id);
    q->wait_and_throw();
  }

  TVMStreamHandle CreateStream(Device dev) final {
    Workspace* w = Workspace::Global();
    return new sycl::queue(w->context(dev.device_id), w->device(dev.device_id),
                           sycl::property::queue::in_order{});
  }

  void FreeStream(Device dev, TVMStreamHandle stream) final {
    delete static_cast<sycl::queue*>(stream);
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) final {
    Workspace* w = Workspace::Global();
    sycl::queue* src = event_src ? static_cast<sycl::queue*>(event_src) : w->queue(dev.device_id);
    sycl::queue* dst = event_dst ? static_cast<sycl::queue*>(event_dst) : w->queue(dev.device_id);
    sycl::event done = src->ext_oneapi_submit_barrier();
    dst->ext_oneapi_submit_barrier({done});
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return Alloc(dev, std::max<size_t>(size, 1));
  }

  void FreeWorkspace(Device dev, void* data) final { Free(dev, data); }

  static OneAPIDeviceAPI* Global() {
    static OneAPIDeviceAPI* inst = new OneAPIDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLCPU) {
      std::memcpy(to, from, size);
      return;
    }
    // USM pointers of any device in the context are valid in one memcpy.
    int id = dev_from.device_type == kDLOneAPI ? dev_from.device_id : dev_to.device_id;
    Workspace* w = Workspace::Global();
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : w->queue(id);
    q->memcpy(to, from, size).wait_and_throw();
  }

 private:
  static void* Alloc(Device dev, size_t nbytes) {
    Workspace* w = Workspace::Global();
    void* ptr = sycl::aligned_alloc_device(kAllocAlignment, nbytes, w->device(dev.device_id),
                                           w->context(dev.device_id));
    TVM_FFI_CHECK(ptr != nullptr, RuntimeError)
        << "oneAPI: cannot allocate " << nbytes << " bytes on device " << dev.device_id;
    return ptr;
  }

  static void Free(Device dev, void* ptr) {
    if (ptr != nullptr) sycl::free(ptr, Workspace::Global()->context(dev.device_id));
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.oneapi", [](ffi::PackedArgs args, ffi::Any* rv) {
    DeviceAPI* ptr = OneAPIDeviceAPI::Global();
    *rv = static_cast<void*>(ptr);
  });
}

}  // namespace oneapi
}  // namespace runtime
}  // namespace tvm
