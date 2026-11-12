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
# pylint: disable=no-member, super-init-not-called

"""OpenCL layout axes.

``sglid`` is a work-item's index within its sub-group.  Placing a coordinate on
it in a register layout states which work-item of the sub-group holds that
value.
"""

import tvm_ffi

from tvm_tirx.tirx.layout import Axis as _Axis

_NAMES = ("sglid",)


class _AxisNamespace:
    def __getattr__(self, name):
        if name in _NAMES:
            return _Axis.get(name)
        raise AttributeError(name)

    def get(self, name):
        if name not in _NAMES:
            raise ValueError(f"Unknown backend axis: {name}")
        return _Axis.get(name)


Axis = _AxisNamespace()


def __getattr__(name):
    return getattr(Axis, name)


__all__ = ["Axis", *_NAMES]


def register():
    tvm_ffi.get_global_func("tirx.opencl.RegisterLayoutAxes")()
