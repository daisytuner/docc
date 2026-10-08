"""ROCm tensor node offloading test suite (see ``_gpu_offload_tensor_nodes_impl``)."""

import pytest

import _gpu_offload_dispatcher_impl as gpu
import _gpu_offload_tensor_nodes_impl as tensor_nodes

pytestmark = pytest.mark.rocm()

tensor_nodes.register(globals(), gpu.ROCM_BACKEND)
