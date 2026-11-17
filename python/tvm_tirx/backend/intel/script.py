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
"""Intel TVMScript operator exports, bound as ``T.intel``.

The Intel tile instructions are published here as ``T.intel.tile.<name>``.  They
run on the ``opencl`` target kind at its sub-group scopes; the namespace says
whose instructions they are.
"""

from tvm_tirx.ir._expose_api import _init_exposed_op_api
from tvm_tirx.script.ir_builder.base import auto_emit_void_call as _auto_emit_void_call

_init_exposed_op_api("tirx.intel", __name__, wrapper=_auto_emit_void_call)
