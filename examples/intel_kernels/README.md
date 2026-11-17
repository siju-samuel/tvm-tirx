<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Intel Xe kernels

Kernels for Intel Xe GPUs written with the `T.intel.tile` instructions. Each module
follows the kernel protocol of [TIRx-kernels](https://github.com/mlc-ai/TIRx-kernels)
(`KERNEL_META`, `CONFIGS`, `get_kernel`, `prepare_data`, `check_correctness`), with
`KERNEL_META["runtime_targets"]` naming the `intel/*` targets it runs on. TIRx-kernels
gates kernels on CUDA architectures only, so they live here until it gains a
non-CUDA entry.

Each module also runs on its own: it checks every config against numpy, and with
`--bench` times its variants interleaved with `tvm_tirx.testing.device_bench`.

```bash
python examples/intel_kernels/gemm.py            # check every config
python examples/intel_kernels/gemm.py --bench "2048^3"
```

| Kernel | What it does |
|---|---|
| [`gemm.py`](gemm.py) | fp16 GEMM with float32 accumulation on DPAS; the leftover tiles of a partly full last wave are split along K |

`tests/python/backend/intel/test_intel_kernels.py` runs the modules' correctness
checks on an Intel GPU.
