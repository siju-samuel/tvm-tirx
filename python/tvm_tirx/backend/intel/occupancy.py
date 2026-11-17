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
"""How many work-groups an Intel Xe GPU runs at once, and how many waves a launch takes.

An Xe core has vector engines, each of which keeps a few hardware threads
resident; a sub-group of 16 work-items is one hardware thread.  How many threads
an engine keeps depends on the registers each one has: an Xe2 engine keeps 8 at
the default 128 registers and 4 at 256.  A work-group's sub-groups all run on
one core, so a core holds as many work-groups as its threads, and its shared
local memory, allow.

A launch of more work-groups than the GPU holds runs in waves, and a last wave
that is only partly full leaves cores idle.  On an Arc Pro B60 the large-tile
GEMM's work-groups (16 sub-groups, 256 registers) fit 2 to a core, 40 in all;
``2048^3`` launches 128 of them, 3.2 waves, so the last wave is a fifth full.
The model predicts the throughput lost to that within 2% of what was measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class XePart:
    """The resources of an Intel Xe GPU that bound how many work-groups it runs.

    Parameters
    ----------
    cores : int
        Xe cores.
    engines_per_core : int
        Vector engines per Xe core.
    threads_per_engine : dict[int, int]
        Hardware threads an engine keeps resident, by registers per thread.
    shared_memory_per_core : int
        Shared local memory a core divides among its work-groups, in bytes.
    """

    cores: int
    engines_per_core: int
    threads_per_engine: dict[int, int]
    shared_memory_per_core: int

    def threads_per_core(self, registers: int) -> int:
        """Hardware threads, so sub-groups, a core keeps resident at ``registers``."""
        if registers not in self.threads_per_engine:
            raise ValueError(
                f"no register mode {registers}; expected one of {sorted(self.threads_per_engine)}"
            )
        return self.engines_per_core * self.threads_per_engine[registers]


# Arc Pro B60 (Battlemage, Xe2): 20 Xe cores of 8 vector engines, which clinfo reports as 160
# compute units.  An Xe2 engine keeps 8 threads at 128 registers and 4 at 256.  The shared local
# memory per core is taken as the 128 KiB one work-group may use, a conservative bound.
ARC_B60 = XePart(
    cores=20, engines_per_core=8, threads_per_engine={128: 8, 256: 4}, shared_memory_per_core=131072
)

_PARTS = {"Arc(TM) Pro B60": ARC_B60}


def part_for_device_name(name):
    """The :class:`XePart` an OpenCL device name belongs to, or None if it is not known.

    Parameters
    ----------
    name : str or None
        The device name OpenCL reports, such as ``tvm_tirx.opencl(0).device_name``.

    Returns
    -------
    part : XePart or None
        The part, or None for one this module has no figures for.
    """
    for needle, part in _PARTS.items():
        if needle in (name or ""):
            return part
    return None


def resident_work_groups(
    part: XePart, *, sub_groups: int, registers: int = 128, shared_memory: int = 0
) -> int:
    """How many work-groups of one kind ``part`` runs at once.

    Parameters
    ----------
    part : XePart
        The GPU.
    sub_groups : int
        Sub-groups per work-group: work-group size divided by the sub-group size.
    registers : int
        Registers per hardware thread the kernel was built with: 128, or 256 in
        large-register mode.
    shared_memory : int
        Shared local memory per work-group, in bytes.

    Returns
    -------
    count : int
        Work-groups resident across the GPU; 0 if one does not fit on a core.
    """
    if sub_groups < 1:
        raise ValueError("a work-group has at least one sub-group")
    per_core = part.threads_per_core(registers) // sub_groups
    if shared_memory > 0:
        per_core = min(per_core, part.shared_memory_per_core // shared_memory)
    return part.cores * per_core


def waves(work_groups: int, resident: int) -> float:
    """How many times over a launch of ``work_groups`` fills ``resident`` slots.

    Parameters
    ----------
    work_groups : int
        Work-groups launched.
    resident : int
        Work-groups the GPU runs at once, from :func:`resident_work_groups`.

    Returns
    -------
    waves : float
        ``work_groups / resident``.
    """
    if resident < 1:
        raise ValueError("no work-group fits; resident must be positive")
    return work_groups / resident


def wave_efficiency(work_groups: int, resident: int) -> float:
    """The fraction of the launch's wave slots that hold a work-group.

    A launch takes ``ceil(waves)`` waves, and every slot of the last one that is
    empty is a core left idle.  With work-groups that all take the same time,
    this is the fraction of full speed the launch can reach.

    Parameters
    ----------
    work_groups : int
        Work-groups launched.
    resident : int
        Work-groups the GPU runs at once.

    Returns
    -------
    efficiency : float
        ``work_groups / (ceil(waves) * resident)``, between 0 and 1.
    """
    if work_groups < 1:
        raise ValueError("a launch has at least one work-group")
    return work_groups / (math.ceil(waves(work_groups, resident)) * resident)
