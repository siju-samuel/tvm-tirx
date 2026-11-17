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
"""Per-kernel OpenCL compiler options.

A target's ``build_options`` apply to every kernel compiled for it.  An
``"opencl"`` entry in a kernel's ``backend_config`` replaces them for that
kernel alone: a kernel that should always get the large register file, say,
while the others keep the target's automatic choice.
"""

from collections.abc import Mapping
from typing import TypedDict

import tvm_ffi


class BackendConfig(TypedDict, total=False):
    """Optional OpenCL overrides.

    ``build_options`` replaces the target's ``build_options`` for the kernel; the
    runtime passes it to the OpenCL compiler when it builds that kernel.
    """

    build_options: str


def validate_backend_config(config):
    """Check the keys and types of an OpenCL backend_config entry.

    Parameters
    ----------
    config : Mapping
        The ``"opencl"`` entry of a ``backend_config``.

    Returns
    -------
    config : dict
        An independently owned copy.
    """
    if not isinstance(config, Mapping):
        raise TypeError("OpenCL backend_config must be a mapping")
    result = {}
    for name, value in config.items():
        if name not in BackendConfig.__annotations__:
            raise ValueError(f"Unknown OpenCL backend_config key: {name!r}")
        if not isinstance(value, str):
            raise TypeError(f"OpenCL backend_config.{name} must be a string")
        if "\n" in value:
            raise ValueError(f"OpenCL backend_config.{name} must be a single line")
        result[name] = value
    return result


@tvm_ffi.register_global_func("opencl.resolve_build_options")
def resolve_build_options(config, target):
    """The compiler options for kernels with ``config``: its entry's, else the target's.

    Parameters
    ----------
    config : str
        The kernels' serialized ``backend_config``; empty when they have none.
    target : tvm_tirx.target.Target
        The target they compile for.

    Returns
    -------
    build_options : str
        The options the runtime passes to the OpenCL compiler for these kernels.
    """
    from tvm_tirx.backend.config import (
        parse_backend_config,  # pylint: disable=import-outside-toplevel
    )

    entry = parse_backend_config(config).get("opencl", {})
    if "build_options" in entry:
        return entry["build_options"]
    return str(target.attrs.get("build_options", ""))
