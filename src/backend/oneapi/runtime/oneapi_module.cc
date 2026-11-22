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
 * \file oneapi_module.cc
 * \brief Runs OpenCL C kernels on oneAPI devices through SYCL, on SYCL queues and USM.
 *
 * The OpenCL code generator's output, one kernel per "// Function:" section with its compiler
 * options on a "// Build options:" line, is built for a device the first time a kernel runs
 * there: to native code with ocloc on a Level Zero device, kept in an on-disk cache, otherwise
 * with DPC++'s kernel compiler.  Kernels launch on the current oneAPI stream, which
 * tvm_ffi.use_torch_stream sets to PyTorch's queue, so they take PyTorch's XPU tensors without
 * a copy and run in order with PyTorch's own work.
 */
#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <sys/stat.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/bytes_io.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <utility>
#include <vector>

#include "../../../runtime/launch_param.h"
#include "../../../runtime/metadata.h"
#include "../../../runtime/pack_args.h"
#include "../../opencl/runtime/source_utils.h"
#include "oneapi_common.h"

namespace tvm {
namespace runtime {
namespace oneapi {

namespace syclex = sycl::ext::oneapi::experimental;

/*!
 * \brief Offline compilation of OpenCL C to a device's native code with libocloc.
 *
 * On a Level Zero device kernels are compiled with ocloc, the way the OpenCL driver compiles
 * them, and loaded as native modules, whose kernels can then be given a scheduling hint (see
 * LoadNativeKernel).  libocloc ships with Intel's GPU driver and has no installed header, so its
 * two entry points are declared here.
 */
class Ocloc {
 public:
  static Ocloc* Get() {
    static Ocloc inst;
    return inst.invoke_ != nullptr ? &inst : nullptr;
  }

  /*! \brief The native binary of ``source`` for the device with PCI id ``device_id``. */
  std::vector<uint8_t> Compile(const std::string& source, const std::string& options,
                               uint32_t device_id) {
    char device[16];
    std::snprintf(device, sizeof(device), "0x%x", device_id);
    std::vector<const char*> argv = {"ocloc", "compile", "-file", "kernel.cl", "-device", device};
    if (!options.empty()) {
      argv.push_back("-options");
      argv.push_back(options.c_str());
    }
    const uint8_t* data = reinterpret_cast<const uint8_t*>(source.c_str());
    uint64_t length = source.size() + 1;
    const char* name = "kernel.cl";
    uint32_t num_outputs = 0;
    uint8_t** outputs = nullptr;
    uint64_t* lengths = nullptr;
    char** names = nullptr;
    int status = invoke_(static_cast<uint32_t>(argv.size()), argv.data(), 1, &data, &length, &name,
                         0, nullptr, nullptr, nullptr, &num_outputs, &outputs, &lengths, &names);
    std::vector<uint8_t> binary;
    std::string log;
    for (uint32_t i = 0; i < num_outputs; ++i) {
      std::string output = names[i];
      if (output.size() > 4 && output.compare(output.size() - 4, 4, ".bin") == 0) {
        binary.assign(outputs[i], outputs[i] + lengths[i]);
      } else if (output.size() > 4 && output.compare(output.size() - 4, 4, ".log") == 0) {
        log.append(reinterpret_cast<const char*>(outputs[i]), lengths[i]);
      }
    }
    free_(&num_outputs, &outputs, &lengths, &names);
    TVM_FFI_CHECK(status == 0 && !binary.empty(), RuntimeError)
        << "ocloc could not compile the kernel for device " << device << ":\n"
        << log;
    return binary;
  }

  /*! \brief The library file ocloc was loaded from, its size and modification time. */
  const std::string& identity() const { return identity_; }

 private:
  using Invoke = int (*)(uint32_t, const char**, uint32_t, const uint8_t**, const uint64_t*,
                         const char**, uint32_t, const uint8_t**, const uint64_t*, const char**,
                         uint32_t*, uint8_t***, uint64_t**, char***);
  using Free = int (*)(uint32_t*, uint8_t***, uint64_t**, char***);

  Ocloc() {
    for (const char* lib : {"libocloc.so", "libocloc.so.1"}) {
      if (void* handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL)) {
        invoke_ = reinterpret_cast<Invoke>(dlsym(handle, "oclocInvoke"));
        free_ = reinterpret_cast<Free>(dlsym(handle, "oclocFreeOutput"));
        if (invoke_ != nullptr && free_ != nullptr) {
          Dl_info info;
          struct stat st;
          if (dladdr(reinterpret_cast<void*>(invoke_), &info) != 0 && info.dli_fname != nullptr &&
              stat(info.dli_fname, &st) == 0) {
            identity_ = std::string(info.dli_fname) + ":" + std::to_string(st.st_size) + ":" +
                        std::to_string(st.st_mtime);
          }
          return;
        }
        invoke_ = nullptr;
      }
    }
  }

  Invoke invoke_ = nullptr;
  Free free_ = nullptr;
  std::string identity_;
};

/*!
 * \brief Native binaries ocloc compiled, on disk, so a kernel compiles once per machine rather
 * than once per process.
 *
 * The OpenCL driver keeps such a cache for the programs it builds, and ocloc does not: on an Arc
 * Pro B60 every process otherwise spent about 0.4 s on each kernel's first launch.  A binary is
 * keyed by all its compilation depends on: the kernel's source and options, the device's PCI
 * id, the driver's version and the ocloc library.  Each file holds its whole key, so a hash
 * collision or a damaged file reads as a miss.  The directory is ``oneapi`` under
 * ``TVM_TIRX_CACHE_DIR``, by default ``$XDG_CACHE_HOME/tvm_tirx`` or ``~/.cache/tvm_tirx``; an
 * empty ``TVM_TIRX_CACHE_DIR`` turns the cache off.  Failing to read or write it only costs a
 * compile.
 */
class BinaryCache {
 public:
  explicit BinaryCache(std::string key) : key_(std::move(key)) {
    std::string dir = Directory();
    if (dir.empty()) return;
    char name[40];
    std::snprintf(name, sizeof(name), "%016llx%016llx.bin",
                  static_cast<unsigned long long>(Fnv1a(key_)),
                  static_cast<unsigned long long>(std::hash<std::string>()(key_)));
    dir_ = dir;
    path_ = dir + "/" + name;
  }

  bool Load(std::vector<uint8_t>* binary) const {
    if (path_.empty()) return false;
    std::ifstream in(path_, std::ios::binary);
    char magic[sizeof(kMagic)];
    uint64_t key_size = 0, size = 0;
    if (!in.read(magic, sizeof(magic)) || std::string(magic, sizeof(magic)) != kMagic ||
        !in.read(reinterpret_cast<char*>(&key_size), sizeof(key_size)) || key_size != key_.size()) {
      return false;
    }
    std::string key(key_size, '\0');
    if (!in.read(key.data(), key_size) || key != key_ ||
        !in.read(reinterpret_cast<char*>(&size), sizeof(size)) || size == 0) {
      return false;
    }
    binary->resize(size);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(binary->data()), size));
  }

  void Store(const std::vector<uint8_t>& binary) const {
    if (path_.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    // Written aside and renamed into place, so a reader never sees half a file.
    std::string tmp = path_ + "." + std::to_string(getpid()) + ".tmp";
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      uint64_t key_size = key_.size(), size = binary.size();
      out.write(kMagic, sizeof(kMagic));
      out.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
      out.write(key_.data(), key_size);
      out.write(reinterpret_cast<const char*>(&size), sizeof(size));
      out.write(reinterpret_cast<const char*>(binary.data()), size);
      if (!out) {
        out.close();
        std::filesystem::remove(tmp, ec);
        return;
      }
    }
    std::filesystem::rename(tmp, path_, ec);
    if (ec) std::filesystem::remove(tmp, ec);
  }

 private:
  static constexpr char kMagic[8] = {'T', 'I', 'R', 'X', 'O', 'C', 'L', '1'};

  static std::string Directory() {
    std::string base;
    if (const char* dir = std::getenv("TVM_TIRX_CACHE_DIR")) {
      if (*dir == '\0') return "";
      base = dir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg != '\0') {
      base = std::string(xdg) + "/tvm_tirx";
    } else if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
      base = std::string(home) + "/.cache/tvm_tirx";
    } else {
      return "";
    }
    return base + "/oneapi";
  }

  static uint64_t Fnv1a(const std::string& text) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : text) hash = (hash ^ c) * 1099511628211ULL;
    return hash;
  }

  std::string key_;
  std::string dir_;
  std::string path_;
};

/*! \brief ``name`` from a native binary, as a SYCL kernel of ``context`` on a Level Zero device. */
inline sycl::kernel LoadNativeKernel(const sycl::context& context, const sycl::device& device,
                                     const std::vector<uint8_t>& binary, const std::string& name) {
  namespace lz = sycl::ext::oneapi::level_zero;
  auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  ze_module_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.format = ZE_MODULE_FORMAT_NATIVE;
  desc.inputSize = binary.size();
  desc.pInputModule = binary.data();
  desc.pBuildFlags = "";
  ze_module_handle_t module = nullptr;
  ze_result_t result = zeModuleCreate(ze_context, ze_device, &desc, &module, nullptr);
  TVM_FFI_CHECK(result == ZE_RESULT_SUCCESS, RuntimeError)
      << "zeModuleCreate failed for kernel " << name << " with code " << result;
  ze_kernel_desc_t kernel_desc = {};
  kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_desc.pKernelName = name.c_str();
  ze_kernel_handle_t kernel = nullptr;
  result = zeKernelCreate(module, &kernel_desc, &kernel);
  TVM_FFI_CHECK(result == ZE_RESULT_SUCCESS, RuntimeError)
      << "zeKernelCreate failed for kernel " << name << " with code " << result;
  // Schedule the kernel's hardware threads round-robin, as the OpenCL driver does by default.
  // Level Zero's default arbitration ran a DPAS GEMM on an Arc Pro B60 at 53 TFLOP/s, against 87
  // for the same native binary round-robin, which is also what the OpenCL runtime got.  A driver
  // without the experimental hint keeps its default.
  ze_scheduling_hint_exp_desc_t hint = {};
  hint.stype = ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC;
  hint.flags = ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN;
  zeKernelSchedulingHintExp(kernel, &hint);
  auto bundle = sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                         sycl::bundle_state::executable>(
      {module, lz::ownership::transfer}, context);
  return sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {bundle, kernel, lz::ownership::transfer}, context);
}

class OneAPIModuleObj : public ffi::ModuleObj {
 public:
  OneAPIModuleObj(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                  ffi::Map<ffi::String, ffi::String> source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {
    TVM_FFI_CHECK(fmt_ == "cl", ValueError)
        << "oneAPI modules run OpenCL C source, got format " << fmt_;
    kernels_source_ = SplitKernels(std::string(code_.data(), code_.size()));
  }

  const char* kind() const final { return "oneapi"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    // The OpenCL module's format: [fmt][fmap][code].  The source map is inspection material
    // and is not saved.
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(result));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (auto it = source_.find(format); it != source_.end()) return (*it).second;
    if (format.empty() || format == "cl") return ffi::String(code_.data(), code_.size());
    return ffi::String();
  }

  /*! \brief The kernel ``name`` built for device ``id``, building it on first use. */
  sycl::kernel GetKernel(int id, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(id, name);
    if (auto it = kernels_.find(key); it != kernels_.end()) return it->second;
    auto src = kernels_source_.find(name);
    TVM_FFI_CHECK(src != kernels_source_.end(), RuntimeError)
        << "oneAPI module has no source for kernel " << name;
    Workspace* w = Workspace::Global();
    sycl::context context = w->context(id);
    sycl::device device = w->device(id);
    std::string option_line = KernelBuildOptions(src->second);
    Ocloc* ocloc = Ocloc::Get();
    if (ocloc != nullptr && device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
        device.has(sycl::aspect::ext_intel_device_id)) {
      uint32_t device_id = device.get_info<sycl::ext::intel::info::device::device_id>();
      BinaryCache cache("device=" + std::to_string(device_id) +
                        "\ndriver=" + device.get_info<sycl::info::device::driver_version>() +
                        "\nocloc=" + ocloc->identity() + "\noptions=" + option_line +
                        "\nsource=" + src->second);
      std::vector<uint8_t> binary;
      if (!cache.Load(&binary)) {
        binary = ocloc->Compile(src->second, option_line, device_id);
        cache.Store(binary);
      }
      sycl::kernel kernel = LoadNativeKernel(context, device, binary, name);
      kernels_.emplace(key, kernel);
      return kernel;
    }
    TVM_FFI_CHECK(device.ext_oneapi_can_build(syclex::source_language::opencl), RuntimeError)
        << "oneAPI device " << id << " cannot build OpenCL C kernels";
    std::vector<std::string> options;
    if (!option_line.empty()) options.push_back(option_line);
    auto bundle = syclex::create_kernel_bundle_from_source(context, syclex::source_language::opencl,
                                                           src->second);
    auto built = syclex::build(bundle, std::vector<sycl::device>{device},
                               syclex::properties{syclex::build_options{options}});
    sycl::kernel kernel = built.ext_oneapi_get_kernel(name);
    kernels_.emplace(key, kernel);
    return kernel;
  }

 private:
  ffi::Bytes code_;
  ffi::String fmt_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  ffi::Map<ffi::String, ffi::String> source_;
  std::unordered_map<std::string, std::string> kernels_source_;
  std::map<std::pair<int, std::string>, sycl::kernel> kernels_;
  std::mutex mutex_;
};

/*! \brief A kernel of a OneAPIModuleObj as a packed function. */
class OneAPIWrappedFunc {
 public:
  void Init(OneAPIModuleObj* m, ffi::ObjectPtr<ffi::Object> sptr, std::string name,
            std::vector<DLDataType> arg_types, const ffi::Array<ffi::String>& launch_param_tags) {
    m_ = m;
    sptr_ = std::move(sptr);
    name_ = std::move(name);
    arg_types_ = std::move(arg_types);
    launch_param_config_.Init(arg_types_.size(), launch_param_tags);
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    int id = CurrentDevice();
    sycl::kernel kernel = m_->GetKernel(id, name_);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    TVM_FFI_CHECK_EQ(wl.dyn_shmem_size, 0U, RuntimeError)
        << "oneAPI kernels do not take dynamic shared memory";
    // SYCL dimension d is OpenCL dimension 2 - d, so TVM's x is SYCL's last.
    sycl::range<3> local(wl.BlockDim(2), wl.BlockDim(1), wl.BlockDim(0));
    sycl::range<3> groups(wl.GridDim(2), wl.GridDim(1), wl.GridDim(0));
    sycl::queue* queue = Workspace::Global()->queue(id);
    queue->submit([&](sycl::handler& cgh) {
      for (size_t i = 0; i < arg_types_.size(); ++i) SetArg(cgh, i, arg_types_[i], void_args[i]);
      cgh.parallel_for(sycl::nd_range<3>(groups * local, local), kernel);
    });
  }

 private:
  static void SetArg(sycl::handler& cgh, size_t i, DLDataType t, void* value) {
    if (t.code == kDLOpaqueHandle) {
      cgh.set_arg(static_cast<int>(i), *static_cast<void**>(value));
      return;
    }
    switch (t.code * 1000 + t.bits) {
      case kDLInt * 1000 + 32:
        cgh.set_arg(static_cast<int>(i), *static_cast<int32_t*>(value));
        return;
      case kDLInt * 1000 + 64:
        cgh.set_arg(static_cast<int>(i), *static_cast<int64_t*>(value));
        return;
      case kDLUInt * 1000 + 32:
        cgh.set_arg(static_cast<int>(i), *static_cast<uint32_t*>(value));
        return;
      case kDLUInt * 1000 + 64:
        cgh.set_arg(static_cast<int>(i), *static_cast<uint64_t*>(value));
        return;
      case kDLFloat * 1000 + 32:
        cgh.set_arg(static_cast<int>(i), *static_cast<float*>(value));
        return;
      case kDLFloat * 1000 + 64:
        cgh.set_arg(static_cast<int>(i), *static_cast<double*>(value));
        return;
      default:
        TVM_FFI_THROW(TypeError) << "oneAPI kernels do not take an argument of type code "
                                 << static_cast<int>(t.code) << " and " << t.bits << " bits";
    }
  }

  OneAPIModuleObj* m_;
  ffi::ObjectPtr<ffi::Object> sptr_;
  std::string name_;
  std::vector<DLDataType> arg_types_;
  LaunchParamConfig launch_param_config_;
};

ffi::Optional<ffi::Function> OneAPIModuleObj::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return std::nullopt;
  FunctionInfo info = opt_info.value();
  std::vector<DLDataType> arg_types(info->arg_types.begin(), info->arg_types.end());
  for (const DLDataType& t : arg_types) TVM_FFI_ICHECK_EQ(t.lanes, 1U);
  OneAPIWrappedFunc f;
  f.Init(this, sptr_to_self, name, arg_types, info->launch_param_tags);
  return PackFuncVoidAddr(f, info->arg_types);
}

ffi::Module OneAPIModuleCreate(ffi::Bytes code, ffi::String fmt,
                               ffi::Map<ffi::String, FunctionInfo> fmap,
                               ffi::Map<ffi::String, ffi::String> source) {
  return ffi::Module(ffi::make_object<OneAPIModuleObj>(std::move(code), std::move(fmt),
                                                       std::move(fmap), std::move(source)));
}

ffi::Module OneAPIModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  return OneAPIModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                            ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.oneapi", OneAPIModuleLoadFromBytes)
      .def("ffi.Module.create.oneapi", OneAPIModuleCreate);
}

}  // namespace oneapi
}  // namespace runtime
}  // namespace tvm
