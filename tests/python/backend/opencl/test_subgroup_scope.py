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
"""OpenCL sub-group execution scopes and the ``sglid`` layout axis."""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.opencl.axis import sglid
from tvm_tirx.backend.opencl.thread_scope import ThreadScope
from tvm_tirx.ir import IntImm
from tvm_tirx.tirx.layout import S, TileLayout


@pytest.mark.parametrize("name", ["kernel", "workgroup", "subgroup", "thread"])
def test_scope_belongs_to_the_opencl_backend(name):
    scope = ThreadScope(name)
    assert str(scope) == f"opencl.{name}"
    assert tvm_tirx.tirx.thread_scope.ThreadScope(f"opencl.{name}").same_as(scope)


def test_scopes_are_ordered_outermost_first():
    from tvm_tirx.tirx import _ffi_api

    ranks = [
        _ffi_api.ThreadScopeGetRank(ThreadScope(n))
        for n in ("kernel", "workgroup", "subgroup", "thread")
    ]
    assert ranks == sorted(ranks)


def test_layout_places_coordinates_on_sglid():
    """Column c of an (8, 16) fragment lives on work-item c, at register row."""
    layout = TileLayout(S[(8, 16) : (1, 1 @ sglid)])
    physical = dict(layout.apply(IntImm("int32", 3), IntImm("int32", 11), shape=[8, 16]))
    assert int(physical["sglid"]) == 11
    assert int(physical["m"]) == 3


if __name__ == "__main__":
    tvm_tirx.testing.main()
