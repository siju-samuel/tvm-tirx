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

``thread_warp_size`` is deliberately left at the ``opencl`` default of 1, even
though Xe parts report sub-groups of 16 and 32.  Declaring the hardware value
today produces OpenCL C that is undefined behavior: ``LowerThreadAllreduce``
switches to its in-warp tail when ``reduce_align <= thread_warp_size``, and that
tail emits ``SyncThread("warp")`` inside a divergent ``if (tid < n)`` region,
which ``CodeGenOpenCL::PrintStorageSync`` lowers to a work-group
``barrier(CLK_LOCAL_MEM_FENCE)``.  A work-group barrier in divergent control flow
is undefined in OpenCL; it happens to compute the right answer on Battlemage but
must not be relied on.  Raising this value requires the OpenCL code generator to
first emit an ``intel_reqd_sub_group_size`` attribute and lower warp-level
synchronization and reductions to sub-group builtins.
"""

from __future__ import annotations

from tvm_tirx.target import register_tag


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
        },
    )


# Conservative baseline for any discrete Xe part: 64 KiB of shared local memory
# is the smallest SLM budget across Xe-HPG and later discrete GPUs.
_register_intel_opencl_tag("intel/xe-opencl", max_threads=1024, shared_mem=65536)

# Arc Pro B60 (Battlemage, BMG): 160 Xe cores, 128 KiB SLM, 1024 threads per
# work-group.  Verified with clinfo on an Arc Pro B60.
_register_intel_opencl_tag("intel/arc-b60-opencl", max_threads=1024, shared_mem=131072)
