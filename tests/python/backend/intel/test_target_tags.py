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
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.target import Target


@pytest.mark.parametrize("tag", ["intel/xe-opencl", "intel/arc-b60-opencl"])
def test_intel_tag_resolves_to_opencl(tag):
    target = Target(tag)
    assert target.kind.name == "opencl"
    assert "intel" in target.keys
    assert "gpu" in target.keys


def test_intel_tag_reports_device_limits():
    """The stock opencl defaults (16 KiB of shared memory, 256 threads) describe
    mobile parts; a discrete Xe part must report its own limits."""
    target = Target("intel/arc-b60-opencl")
    assert target.attrs["max_shared_memory_per_block"] == 131072
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["max_num_threads"] == 1024

    stock = Target("opencl")
    assert target.attrs["max_shared_memory_per_block"] > stock.attrs["max_shared_memory_per_block"]
    assert target.attrs["max_threads_per_block"] > stock.attrs["max_threads_per_block"]


def test_intel_tag_keeps_default_warp_size():
    """Declaring the hardware sub-group size (16) would push LowerThreadAllreduce
    into its in-warp tail, which emits a work-group barrier inside divergent
    control flow -- undefined behavior in OpenCL.  See target_tags docstring."""
    assert Target("intel/arc-b60-opencl").attrs["thread_warp_size"] == 1
    assert Target("intel/xe-opencl").attrs["thread_warp_size"] == 1


def test_intel_backend_is_autoloaded():
    import tvm_tirx.backend as backend

    assert backend.is_loaded("intel")


if __name__ == "__main__":
    tvm_tirx.testing.main()
