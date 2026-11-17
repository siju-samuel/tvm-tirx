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
"""Device timers, which measure work on the device rather than on the host."""

from __future__ import annotations

import tvm_ffi
from tvm_ffi import Object

from . import _ffi_api


@tvm_ffi.register_object("runtime.Timer")
class Timer(Object):
    """A timer for one span of work on one device.

    Make one with :func:`device_timer`, then call :meth:`start` before the work,
    :meth:`stop` after it, and :meth:`elapsed_ns` to read it; each once.  The
    work is measured on the device, so host launch overhead is left out.  The
    OpenCL timer sums the run time of each kernel launched in the span.
    """

    def start(self) -> None:
        """Start timing the work submitted from now on."""
        _ffi_api.TimerStart(self)

    def stop(self) -> None:
        """Stop timing; the work submitted so far is the span measured."""
        _ffi_api.TimerStop(self)

    def elapsed_ns(self) -> int:
        """Wait for the span to finish and return its length in nanoseconds.

        Returns
        -------
        elapsed : int
            The time the span took on the device, in nanoseconds.
        """
        return int(_ffi_api.TimerSyncAndGetElapsedNanos(self))


def device_timer(device) -> Timer | None:
    """A timer for ``device``, if its runtime has one.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device
        The device whose work to time.

    Returns
    -------
    timer : Timer or None
        A new timer, or None when the device's runtime registers none or is not
        in this build.
    """
    name = type(device)._DEVICE_TYPE_TO_NAME.get(device.dlpack_device_type())
    make = tvm_ffi.get_global_func(f"runtime.timer.{name}", allow_missing=True)
    return None if make is None else make(device)
