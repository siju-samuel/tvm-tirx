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
"""Interleaved timing of kernels on any device with a TIRx runtime timer.

:mod:`tvm_tirx.testing.bench` times through torch's CUDA events.  This module
times through the TIRx runtime's own device timers, so it also works where torch
cannot see the device, such as an Intel GPU reached through OpenCL, and it needs
neither torch nor Triton.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tvm_tirx.runtime import device_timer

_SEPARATOR_SIZE = 256


@dataclass(frozen=True)
class Measurement:
    """The per-call time of one function in each round, in microseconds."""

    samples_us: tuple[float, ...]

    @property
    def median_us(self) -> float:
        """The median per-call time over the rounds, in microseconds."""
        return statistics.median(self.samples_us)


def bench_interleaved(
    funcs: Mapping[str, Callable[[], None]],
    device,
    *,
    rounds: int = 10,
    number: int = 20,
    warmup: int = 2,
    timer: str = "host",
    between: Callable[[], None] | None = None,
) -> dict[str, Measurement]:
    """Time launch closures in turn, round after round.

    One measurement on a shared GPU is dominated by clock ramp-up and by
    whatever else runs on it.  Running every candidate in turn within each
    round, and taking each one's median over the rounds, compares them under
    the same conditions.

    Parameters
    ----------
    funcs : Mapping[str, Callable[[], None]]
        No-argument closures that launch the work to compare, by name.
    device : tvm_tirx.runtime.Device
        The device the work runs on.
    rounds : int
        How many times to time each closure; each round gives one sample.
    number : int
        Timed calls per round; a sample is their mean.
    warmup : int
        Untimed calls before each round's timed ones.
    timer : {"host", "device"}
        ``"host"`` (the default) measures wall time from the first timed launch
        until the device is idle: the throughput of back-to-back launches, gaps
        included.  ``"device"`` uses the device's runtime timer instead, which
        leaves out the host; the OpenCL one adds up each kernel's profiled run
        time.  The two need not agree: for back-to-back launches of one kernel
        on an Arc Pro B60, the OpenCL sum came out about 10% above the wall
        time.  Compare candidates with one timer, not across timers.
    between : Callable[[], None], optional
        A closure launched after every call, such as a small kernel, so that
        each candidate runs as one operation among others rather than back to
        back.  Back-to-back launches of one kernel can overlap (their run times
        add up to more than the wall time, as above), so one launch can hide the
        last one's tail, which in a model, where other work comes between, it
        cannot.  Two plans of a paged decode attention 3% apart back to back on
        an Arc Pro B60 were 28% apart with a one-work-group kernel between.
        Each sample includes ``between``.  Its time alone is no measure of what
        it adds: alone, the B60's one-work-group kernel took 5.3 us a launch,
        the host's launch rate, but after a 165 us kernel it added 1.8 us.

    Returns
    -------
    results : dict[str, Measurement]
        Each closure's samples, in the order of ``funcs``.
    """
    if timer not in ("device", "host"):
        raise ValueError(f"timer must be 'device' or 'host', got {timer!r}")
    if rounds < 1 or number < 1 or warmup < 0:
        raise ValueError("rounds and number must be positive and warmup not negative")
    if timer == "device" and device_timer(device) is None:
        raise ValueError(f"{device} has no runtime timer; use timer='host'")

    def per_call_us(fn):
        for _ in range(warmup):
            fn()
        device.sync()
        if timer == "device":
            clock = device_timer(device)
            clock.start()
            for _ in range(number):
                fn()
            clock.stop()
            elapsed = clock.elapsed_ns() * 1e-9
        else:
            start = time.perf_counter()
            for _ in range(number):
                fn()
            device.sync()
            elapsed = time.perf_counter() - start
        return elapsed / number * 1e6

    def followed(fn):
        def call():
            fn()
            between()

        return call

    samples = {name: [] for name in funcs}
    for _ in range(rounds):
        for name, fn in funcs.items():
            samples[name].append(per_call_us(fn if between is None else followed(fn)))
    return {name: Measurement(tuple(values)) for name, values in samples.items()}


def separator(device, target=None) -> Callable[[], None]:
    """A launch of a one-work-group kernel on a GPU, to pass as ``between``.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device
        The GPU.
    target : str or tvm_tirx.target.Target, optional
        The target to build the kernel for; by default the one detected for
        ``device``.

    Returns
    -------
    launch : Callable[[], None]
        A closure that launches the kernel once.  Each launch adds 1 to the
        elements of a small buffer the closure owns.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    import tvm_tirx
    from tvm_tirx.script import ir as I
    from tvm_tirx.script import tirx as T
    from tvm_tirx.target import Target

    n = _SEPARATOR_SIZE

    @I.ir_module
    class Separator:
        @T.function
        def main(A: T.Tensor((n,), "float32")):
            with T.launch_thread("blockIdx.x", 1) as _:
                with T.launch_thread("threadIdx.x", n) as tx:
                    A[tx] = A[tx] + T.float32(1)

    fun = tvm_tirx.compile(Separator, target=target or Target.from_device(device))["main"]
    buffer = tvm_tirx.runtime.tensor(np.zeros(n, np.float32), device)

    def launch():
        fun(buffer)

    launch.buffer = buffer
    return launch
