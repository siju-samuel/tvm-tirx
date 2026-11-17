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
"""Intel Xe tile instructions, published as ``T.intel.tile``.

They run on the ``opencl`` target kind, which is how Intel GPUs are reached, at
its sub-group scopes; the namespace says whose instructions they are, as
``T.ptx`` does under the ``cuda`` kind.  Each one checks that its target is an
Intel part before lowering.
"""

from . import block_io, dpas, ewise

__all__ = ["block_io", "dpas", "ewise"]
