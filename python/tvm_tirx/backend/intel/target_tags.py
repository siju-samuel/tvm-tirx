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
"""Intel Xe GPU target tags.

Intel discrete GPUs run through the ``opencl`` target kind: the Intel Compute
Runtime (NEO) compiles the generated OpenCL C with IGC.  The ``opencl`` kind's
defaults are tuned for mobile parts -- 16 KiB of shared memory and 256 threads
per block -- which badly under-describes a discrete Xe part, so these tags carry
the real device limits.

``thread_warp_size`` is 16, the SIMD width Xe parts use for compute.  Declaring
it is what puts reductions on ``sub_group_reduce``-style shuffles instead of a
shared-local-memory tree: ``LowerThreadAllreduce`` only takes its warp path for a
target whose ``thread_warp_size`` exceeds 1, and ``CodeGenOpenCL`` only pins
``intel_reqd_sub_group_size`` and emits ``sub_group_barrier`` for such a target.
Leaving it at the ``opencl`` default of 1 means "width unknown" and keeps the
shared-memory reduction.

Xe parts also support sub-groups of 8 and 32.  16 is chosen because it is the
width IGC picks for compute kernels by default, so pinning it with
``intel_reqd_sub_group_size`` agrees with what the hardware would have done
anyway.  The value has to be declared rather than queried: it is baked into the
generated code at compile time, while the device query in
``opencl_device_api.cc`` reports what the driver offers at run time.
"""

from __future__ import annotations

from tvm_tirx.target import register_tag

# SIMD width Xe parts use for compute kernels.  See the module docstring.
_XE_SUB_GROUP_SIZE = 16


def _register_intel_opencl_tag(name: str, *, max_threads: int, shared_mem: int) -> None:
    register_tag(
        name,
        {
            "kind": "opencl",
            "device": "intel_xe",
            "keys": ["intel", "opencl", "gpu"],
            "max_threads_per_block": max_threads,
            "max_num_threads": max_threads,
            "max_shared_memory_per_block": shared_mem,
            "thread_warp_size": _XE_SUB_GROUP_SIZE,
        },
    )


# Conservative baseline for any discrete Xe part: 64 KiB of shared local memory
# is the smallest SLM budget across Xe-HPG and later discrete GPUs.
_register_intel_opencl_tag("intel/xe-opencl", max_threads=1024, shared_mem=65536)

# Arc Pro B60 (Battlemage, BMG): 160 Xe cores, 128 KiB SLM, 1024 threads per
# work-group.  Verified with clinfo on an Arc Pro B60.
_register_intel_opencl_tag("intel/arc-b60-opencl", max_threads=1024, shared_mem=131072)
