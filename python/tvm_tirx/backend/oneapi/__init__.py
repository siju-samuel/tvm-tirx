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
"""oneAPI backend hooks: kDLOneAPI devices, the kind PyTorch's XPU tensors are on.

There is no oneapi target kind.  The kernels are OpenCL C from the opencl code
generator, so they go through its lowering, sub-group proofs and Intel tile
instructions unchanged.  A target of kind ``opencl`` whose ``target_device_type``
is kDLOneAPI (14) gets a module that runs them through SYCL instead: on unified
shared memory and SYCL queues, so a kernel takes PyTorch's XPU tensors without a
copy and, under ``tvm_ffi.use_torch_stream``, runs on PyTorch's queue.  The
``intel/*-oneapi`` tags are such targets.

The SYCL runtime is the ``tvm_runtime_oneapi`` library, built with
``TVM_TIRX_USE_ONEAPI=ON`` and icpx.  Without it, compiling still works and the
module keeps its source for export.
"""

from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from tvm_tirx.base import _LOADED_LIBS

ONEAPI_DEVICE_TYPE = 14  # kDLOneAPI


def _detect_target_from_device(dev):
    from tvm_tirx.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "opencl",
            "target_device_type": ONEAPI_DEVICE_TYPE,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def register_backend():
    """Register oneAPI-owned Python semantics and load the SYCL runtime if it was built."""
    import tvm_tirx.backend as backend  # pylint: disable=import-outside-toplevel
    from tvm_tirx.target.detect_target import (  # pylint: disable=import-outside-toplevel
        register_device_target_detector,
    )

    backend.load("opencl")
    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _LOADED_LIBS["tvm_runtime_oneapi"] = load_lib_ctypes(
            package="tvm_tirx",
            target_name="tvm_runtime_oneapi",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass
    register_device_target_detector("oneapi", _detect_target_from_device)


def runtime_enabled():
    """Whether the SYCL runtime is loaded, so oneAPI modules can run kernels.

    Returns
    -------
    enabled : bool
        True when ``tvm_runtime_oneapi`` was built and loaded.
    """
    import tvm_ffi  # pylint: disable=import-outside-toplevel

    return tvm_ffi.get_global_func("ffi.Module.create.oneapi", allow_missing=True) is not None


__all__ = ["ONEAPI_DEVICE_TYPE", "register_backend", "runtime_enabled"]
