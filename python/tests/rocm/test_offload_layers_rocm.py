"""ROCm layer-offloading test suite (see ``_gpu_offload_layer_impl``)."""

import pytest

import _gpu_offload_dispatcher_impl as gpu
import _gpu_offload_layer_impl as layers

pytestmark = pytest.mark.rocm()

layers.register(globals(), gpu.ROCM_BACKEND)
