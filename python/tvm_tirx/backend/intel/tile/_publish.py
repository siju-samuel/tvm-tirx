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
"""Expose an Intel tile instruction's own constructor in TVMScript."""

from importlib import import_module


def publish(name, factory):
    """Bind ``factory`` where ``T.intel.tile.<name>`` resolves.

    Registering an instruction also exposes a generic op constructor under its
    name, and that one hands ``scope="subgroup"`` to the scope enum unqualified.
    The instruction's factory qualifies it to ``opencl.subgroup`` and validates
    the operands, so it replaces the generic one in the script namespace, wrapped
    to emit its call the way that namespace's other ops do.
    """
    from tvm_tirx.script.ir_builder.base import (  # pylint: disable=import-outside-toplevel
        auto_emit_void_call,
    )

    script = import_module("tvm_tirx.backend.intel.script")
    setattr(script.tile, name, auto_emit_void_call(factory))
