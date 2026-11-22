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
"""Device timers, and the interleaved benchmark built on them."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.runtime import Timer, device_timer
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.testing import env
from tvm_tirx.testing.device_bench import bench_interleaved, separator


def test_cpu_has_no_device_timer():
    assert device_timer(tvm_tirx.cpu(0)) is None


def test_device_timing_needs_a_device_timer():
    with pytest.raises(ValueError, match="has no runtime timer"):
        bench_interleaved({"a": lambda: None}, tvm_tirx.cpu(0), timer="device")


def test_rounds_run_every_candidate_in_turn():
    calls = []
    results = bench_interleaved(
        {"a": lambda: calls.append("a"), "b": lambda: calls.append("b")},
        tvm_tirx.cpu(0),
        rounds=3,
        number=2,
        warmup=1,
        timer="host",
    )
    assert list(results) == ["a", "b"]
    assert all(len(m.samples_us) == 3 for m in results.values())
    # Each round runs a's warmup and timed calls, then b's.
    assert calls == ["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 3


def test_between_follows_every_call_and_counts_in_the_samples(monkeypatch):
    """A fake clock that each closure advances makes the samples exact."""
    from tvm_tirx.testing import device_bench

    clock = [0.0]
    calls = []
    monkeypatch.setattr(device_bench.time, "perf_counter", lambda: clock[0])

    def tick(name, seconds):
        def fn():
            calls.append(name)
            clock[0] += seconds

        return fn

    results = bench_interleaved(
        {"a": tick("a", 3e-6)},
        tvm_tirx.cpu(0),
        rounds=2,
        number=2,
        warmup=1,
        between=tick("x", 1e-6),
    )
    assert calls == ["a", "x"] * 3 * 2
    assert results["a"].samples_us == pytest.approx((4.0, 4.0))


N = 1 << 20


def _busy_kernel(target):
    """Each work-item runs a long chain of dependent multiply-adds."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", N // 256) as bx:
                with T.launch_thread("threadIdx.x", 256) as tx:
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = A[bx * 256 + tx]
                    for _ in range(512):
                        acc[0] = acc[0] * T.float32(0.999) + T.float32(0.001)
                    B[bx * 256 + tx] = acc[0]

    return tvm_tirx.compile(Module, target=target)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_opencl_timer_measures_the_kernels():
    def run_and_check():
        dev = tvm_tirx.opencl(0)
        fun = _busy_kernel(Target.from_device(dev))
        args = [tvm_tirx.runtime.tensor(np.ones(N, np.float32), dev) for _ in range(2)]
        fun["main"](*args)
        dev.sync()

        timer = device_timer(dev)
        assert isinstance(timer, Timer)
        timer.start()
        for _ in range(10):
            fun["main"](*args)
        timer.stop()
        device_ns = timer.elapsed_ns()
        assert device_ns > 0

        # The two timers measure differently, so only check that they agree on the scale.
        device = bench_interleaved(
            {"busy": lambda: fun["main"](*args)}, dev, rounds=3, number=10, timer="device"
        )
        host = bench_interleaved({"busy": lambda: fun["main"](*args)}, dev, rounds=3, number=10)
        ratio = device["busy"].median_us / host["busy"].median_us
        assert 0.5 < ratio < 2, ratio
        np.testing.assert_allclose(args[1].numpy(), args[1].numpy()[0], rtol=1e-6)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_separator_runs_between_calls():
    def run_and_check():
        dev = tvm_tirx.opencl(0)
        launch = separator(dev)
        results = bench_interleaved(
            {"noop": lambda: None}, dev, rounds=2, number=3, warmup=1, between=launch
        )
        # Two rounds of a warmup call and three timed ones, each followed by the separator.
        np.testing.assert_array_equal(launch.buffer.numpy(), 2 * (1 + 3))
        assert len(results["noop"].samples_us) == 2

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
