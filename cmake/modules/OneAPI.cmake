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

# The oneAPI runtime runs the OpenCL code generator's kernels through SYCL, on kDLOneAPI
# devices: unified shared memory and SYCL queues, which is what PyTorch's XPU tensors and streams
# are.  It is SYCL C++, so it is built with a SYCL compiler (icpx) as its own library, while the
# rest of TIRx keeps the project's compiler.  At run time it uses the SYCL runtime already in the
# process, such as the one PyTorch's XPU build brings, so build it with an icpx of the same
# release.  Intel's math libraries are linked statically (-static-intel), so the SYCL runtime is
# its only oneAPI dependency.  On Level Zero devices it compiles kernels to native code with the
# GPU driver's libocloc, loaded at run time, and loads them through the Level Zero loader, so it
# also needs the Level Zero headers to build.
if(TVM_TIRX_USE_ONEAPI)
  if(TVM_TIRX_USE_ONEAPI MATCHES ${IS_TRUE_PATTERN})
    find_program(TVM_TIRX_ONEAPI_ICPX icpx HINTS ENV CMPLR_ROOT PATH_SUFFIXES bin)
  else()
    set(TVM_TIRX_ONEAPI_ICPX "${TVM_TIRX_USE_ONEAPI}")
  endif()
  if(NOT TVM_TIRX_ONEAPI_ICPX OR NOT EXISTS "${TVM_TIRX_ONEAPI_ICPX}")
    message(FATAL_ERROR "TVM_TIRX_USE_ONEAPI needs icpx: put it on PATH, set CMPLR_ROOT, "
                        "or pass its path as TVM_TIRX_USE_ONEAPI")
  endif()
  message(STATUS "Build the oneAPI (SYCL) runtime with ${TVM_TIRX_ONEAPI_ICPX}")

  file(GLOB RUNTIME_ONEAPI_SRCS CONFIGURE_DEPENDS src/backend/oneapi/runtime/*.cc)
  set(_oneapi_lib
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}tvm_runtime_oneapi${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(_oneapi_includes "$<TARGET_PROPERTY:tvm_runtime_objs,INCLUDE_DIRECTORIES>")
  set(_oneapi_defines "$<TARGET_PROPERTY:tvm_runtime_objs,COMPILE_DEFINITIONS>")
  add_custom_command(
    OUTPUT "${_oneapi_lib}"
    COMMAND "${TVM_TIRX_ONEAPI_ICPX}" -fsycl -shared -fPIC -O2 -std=c++20 -static-intel
            "$<$<BOOL:${_oneapi_includes}>:-I$<JOIN:${_oneapi_includes},;-I>>"
            "$<$<BOOL:${_oneapi_defines}>:-D$<JOIN:${_oneapi_defines},;-D>>"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            ${RUNTIME_ONEAPI_SRCS}
            "$<TARGET_FILE:tvm_runtime>" "$<TARGET_FILE:tvm_ffi_shared>" -lze_loader -ldl
            "-Wl,-rpath,\$ORIGIN" "-Wl,-rpath,\$ORIGIN/../../tvm_ffi/lib" -Wl,--no-undefined
            -o "${_oneapi_lib}"
    DEPENDS ${RUNTIME_ONEAPI_SRCS} tvm_runtime
    COMMENT "Building the oneAPI runtime with icpx"
    COMMAND_EXPAND_LISTS
    VERBATIM)
  add_custom_target(tvm_runtime_oneapi ALL DEPENDS "${_oneapi_lib}")
  install(FILES "${_oneapi_lib}" DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx)
endif(TVM_TIRX_USE_ONEAPI)
