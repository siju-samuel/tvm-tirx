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

``build_options`` lets the compiler choose each kernel's register file.  Xe
work-items have 128 registers by default or 256 in large-register mode, which
halves how many threads a core keeps resident.  A kernel whose live state does
not fit in 128 spills, and a large matrix tile then runs over 30x slower; but
forcing 256 everywhere costs a kernel that fits a third of its speed.  IGC's
automatic mode takes a larger file only for a kernel that would otherwise
spill, which on an Arc Pro B60 matched the faster fixed mode for both kinds of
kernel.

Importing this module also registers a device detector, so
``Target.from_device`` on a tagged part returns its tag rather than a target
built from the device query, whose ``thread_warp_size`` of 1 would turn the
sub-group lowering off.  ``device_tag`` names the tag of an OpenCL device.
"""

from __future__ import annotations

from tvm_tirx.target import register_tag

# SIMD width Xe parts use for compute kernels.  See the module docstring.
_XE_SUB_GROUP_SIZE = 16
# Let IGC pick 128 or 256 registers per kernel.  See the module docstring.
_XE_BUILD_OPTIONS = "-cl-intel-enable-auto-large-GRF-mode"


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
            "build_options": _XE_BUILD_OPTIONS,
        },
    )


# Conservative baseline for any discrete Xe part: 64 KiB of shared local memory
# is the smallest SLM budget across Xe-HPG and later discrete GPUs.
_register_intel_opencl_tag("intel/xe-opencl", max_threads=1024, shared_mem=65536)

# Arc Pro B60 (Battlemage, BMG): 20 Xe2 cores, 128 KiB SLM, 1024 threads per
# work-group.  Verified with clinfo on an Arc Pro B60, whose 160 compute units are
# the cores' vector engines, eight per core.
_register_intel_opencl_tag("intel/arc-b60-opencl", max_threads=1024, shared_mem=131072)

# How each tagged part names itself through OpenCL.
_DEVICE_NAMES = {
    "intel/arc-b60-opencl": ("Arc(TM) Pro B60",),
}


def tag_for_device_name(name):
    """The ``intel/*`` tag of the part an OpenCL device name belongs to.

    Parameters
    ----------
    name : str or None
        The device name OpenCL reports, such as ``tvm_tirx.opencl(0).device_name``.

    Returns
    -------
    tag : str or None
        The tag, or None for a part that has none.
    """
    for tag, needles in _DEVICE_NAMES.items():
        if any(needle in (name or "") for needle in needles):
            return tag
    return None


def device_tag(index=0):
    """The ``intel/*`` tag of an OpenCL device.

    Parameters
    ----------
    index : int, optional
        The OpenCL device index.

    Returns
    -------
    tag : str or None
        The tag, or None when there is no such device, no OpenCL runtime in this
        build, or the device is not a tagged part.
    """
    import tvm_tirx  # pylint: disable=import-outside-toplevel

    try:
        device = tvm_tirx.opencl(index)
        if not device.exist:
            return None
        return tag_for_device_name(device.device_name)
    except Exception:  # pylint: disable=broad-except
        # No OpenCL runtime in this build, or no driver on this machine.
        return None


def _install_device_detector():
    """Detect a tagged Intel part as its tag, so it gets the sub-group width and limits.

    The generic OpenCL detector builds a target from what the device reports, which
    leaves ``thread_warp_size`` at 1 and so keeps the sub-group lowering off.  Any
    other device still goes to that detector, which exists by now: the Intel
    backend loads the opencl backend before it imports this module.
    """
    from tvm_tirx.target import Target  # pylint: disable=import-outside-toplevel
    from tvm_tirx.target.detect_target import (  # pylint: disable=import-outside-toplevel
        SUPPORTED_DEVICE,
        register_device_target_detector,
    )

    generic = SUPPORTED_DEVICE.get("opencl")

    def detect(dev):
        tag = tag_for_device_name(dev.device_name)
        if tag is not None:
            return Target(tag)
        if generic is None:
            raise ValueError(f"No target detector for OpenCL device {dev.device_name!r}")
        return generic(dev)

    register_device_target_detector("opencl", detect)


_install_device_detector()
