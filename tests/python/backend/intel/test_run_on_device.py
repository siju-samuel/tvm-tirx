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
"""Run generated kernels on an Intel Xe GPU through the OpenCL runtime."""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

TARGET = "intel/arc-b60-opencl"


def _has_intel_opencl_device() -> bool:
    if not env.has_opencl():
        return False
    name = tvm_tirx.opencl(0).device_name
    return bool(name) and "Intel" in name


requires_intel_gpu = pytest.mark.skipif(
    not _has_intel_opencl_device(), reason="need an Intel OpenCL GPU"
)


@pytest.mark.gpu
@requires_intel_gpu
def test_add_one():
    import numpy as np

    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0

    fun = tvm_tirx.compile(Module, target=TARGET)
    a_np = np.random.uniform(size=(n,)).astype("float32")

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(np.zeros((n,), dtype="float32"), dev)
        fun["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np + 1.0, rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_intel_gpu
def test_crossthread_reduction():
    """Exercises shared local memory and barriers: the reduction lowers to a
    ``__local`` tree reduction because the OpenCL backend has no sub-group path."""
    import numpy as np

    nthd = 64
    n = T.dynamic("n", "int32")
    m = T.dynamic("m", "int32")

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n, m), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n) as i:
                with T.launch_thread("threadIdx.x", nthd) as m_0:
                    partial = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                    reduced = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                    partial[0] = T.float32(0)
                    for m_1 in range((m + nthd - 1) // nthd):
                        if m_0 * ((m + nthd - 1) // nthd) + m_1 < m:
                            partial[0] = partial[0] + A[i, m_0 * ((m + nthd - 1) // nthd) + m_1]
                    T.gpu_thread_allreduce(
                        T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                        (T.float32(0),),
                        (partial[0],),
                        True,
                        (reduced[0],),
                        (m_0,),
                    )
                    if m_0 == 0:
                        B[i] = reduced[0]

    fun = tvm_tirx.compile(Module, target=TARGET)
    rows, cols = 128, 1021  # cols is deliberately not a multiple of nthd
    a_np = np.random.uniform(size=(rows, cols)).astype("float32")

    def run_and_check():
        dev = tvm_tirx.opencl(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(np.zeros((rows,), dtype="float32"), dev)
        fun["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np.sum(axis=1), rtol=1e-4, atol=1e-4)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
