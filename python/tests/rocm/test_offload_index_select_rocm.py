"""ROCm index_select offloading test suite (see ``_gpu_offload_index_select_impl``)."""

import pytest

import _gpu_offload_dispatcher_impl as gpu
import _gpu_offload_index_select_impl as index_select

pytestmark = pytest.mark.rocm()

index_select.register(globals(), gpu.ROCM_BACKEND)
