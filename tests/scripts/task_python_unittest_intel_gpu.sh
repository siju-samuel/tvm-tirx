#!/usr/bin/env bash
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

# Run the Intel GPU tests, and the OpenCL and shared GPU tests they build on, on a
# machine with an Intel GPU and its OpenCL runtime.
#
# Build with the OpenCL runtime enabled first: tests/scripts/task_build_gpu_other.sh,
# or for an editable install
#     CMAKE_ARGS="-DTVM_TIRX_USE_OPENCL=ON" uv sync --no-default-groups --group test
#
# Upstream CI has no Intel runner yet; a job on one would call this script.
set -euxo pipefail

# The device tests skip themselves when no Intel GPU is visible, so a runner without
# one would pass while testing nothing.  Refuse to run there instead.
uv run --no-sync python - <<'PY'
import sys

import tvm_tirx
from tvm_tirx.backend.intel.target_tags import device_tag

device = tvm_tirx.opencl(0)
if not device.exist or "Intel" not in (device.device_name or ""):
    sys.exit("No Intel OpenCL GPU is visible; the Intel GPU tests would only be skipped.")
# An untagged part is tested with the B60's tag.
print(f"Intel GPU tests on {device.device_name}, as {device_tag() or 'intel/arc-b60-opencl'}")
PY

if [ "$#" -eq 0 ]; then
    set -- tests/python/backend/intel tests/python/backend/opencl tests/python/codegen/gpu_common
fi
# The verifier and codegen tests need no device; run them alongside the gpu tests.
export TVM_TEST_MARKER="gpu or not gpu"
./tests/scripts/task_python_unittest.sh "$@"
