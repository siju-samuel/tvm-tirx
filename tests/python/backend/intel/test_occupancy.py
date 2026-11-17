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
"""The occupancy and wave model of Intel Xe GPUs."""

import pytest

import tvm_tirx.testing
from tvm_tirx.backend.intel.occupancy import (
    ARC_B60,
    part_for_device_name,
    resident_work_groups,
    wave_efficiency,
    waves,
)


def test_large_tile_gemm_fits_two_work_groups_per_b60_core():
    """16 sub-groups of 256 registers: a core's 32 threads hold two work-groups."""
    assert resident_work_groups(ARC_B60, sub_groups=16, registers=256) == 40
    assert resident_work_groups(ARC_B60, sub_groups=16, registers=128) == 80


def test_shared_memory_can_bound_residency():
    assert resident_work_groups(ARC_B60, sub_groups=4, shared_memory=65536) == 40
    assert resident_work_groups(ARC_B60, sub_groups=4, shared_memory=200000) == 0


@pytest.mark.parametrize(
    "m, n, expected_waves, expected_efficiency",
    [
        # 256 x 128 work-group tiles, 40 resident: what was measured on a B60.
        (2048, 2048, 3.2, 0.8),
        (2560, 2048, 4.0, 1.0),
        (1792, 2048, 2.8, 2.8 / 3),
    ],
)
def test_waves_of_the_large_tile_gemm(m, n, expected_waves, expected_efficiency):
    work_groups = (m // 256) * (n // 128)
    assert waves(work_groups, 40) == pytest.approx(expected_waves)
    assert wave_efficiency(work_groups, 40) == pytest.approx(expected_efficiency)


def test_unknown_register_mode_is_rejected():
    with pytest.raises(ValueError, match="no register mode 512"):
        resident_work_groups(ARC_B60, sub_groups=16, registers=512)


def test_device_name_selects_its_part():
    assert part_for_device_name("Intel(R) Arc(TM) Pro B60 Graphics") is ARC_B60
    assert part_for_device_name("QUALCOMM Adreno(TM) 740") is None


if __name__ == "__main__":
    tvm_tirx.testing.main()
