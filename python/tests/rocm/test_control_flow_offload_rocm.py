"""ROCm control-flow offload test suite.

A thin binding of the backend-agnostic scenarios in
``_gpu_control_flow_offload_impl`` to the ROCm target. Every test is generated
by :func:`register` and carries the module-level ``rocm`` marker.
"""

import pytest

import _gpu_control_flow_offload_impl as cf

pytestmark = pytest.mark.rocm()

cf.register(globals(), "rocm", "cupy")
