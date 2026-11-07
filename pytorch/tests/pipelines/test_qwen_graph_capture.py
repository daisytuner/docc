"""Capture the compiled Qwen2 MLP into a HIP graph and replay it.

The compiled call must run entirely on the caller's stream and must not run
synchronous allocations inside the captured region: kernels launch on the stream
set through ``__daisy_set_stream``, device scratch uses the stream-ordered
``hipMallocAsync``/``hipFreeAsync`` (``rocm_async_alloc=True``), and the output
is a persistent torch buffer with a stable address.

Run with ``pytest --target=rocm tests/pipelines/test_qwen_graph_capture.py``.
"""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from docc.compiler import DoccOptions
from docc.pytorch.pytorch_program import PyTorchProgram

# Qwen2.5-1.5B MLP dimensions. gate_up_proj fuses gate and up.
HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960
RELATIVE_ERROR: dict[torch.dtype, float] = {
    torch.bfloat16: 0.2,
    torch.float32: 5e-7,
}


class _StaticMLP(torch.nn.Module):
    """Qwen2 MLP as a pure function of (x, w_gate_up, w_down)."""

    def forward(self, x, w_gate_up, w_down):
        gate_up = F.linear(x, w_gate_up)
        d = gate_up.shape[-1] // 2
        h = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        return F.linear(h, w_down)


def _async_alloc_works() -> bool:
    """Probe hipMallocAsync; it fails on some GPUs (for example gfx90a)."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    ptr = ctypes.c_void_p()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    err = hip.hipMallocAsync(ctypes.byref(ptr), ctypes.c_size_t(1 << 20), stream)
    if err != 0:
        return False
    hip.hipFreeAsync(ptr, stream)
    torch.cuda.synchronize()
    return True


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out32, ref32 = out.to(torch.float32), ref.to(torch.float32)
    return float(
        torch.linalg.vector_norm(out32 - ref32) / torch.linalg.vector_norm(ref32)
    )


@pytest.mark.supported_targets("rocm")
@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32"),
    ],
)
def test_qwen2_mlp_hip_graph_capture(
    target: str, tokens: int, dtype: torch.dtype
) -> None:
    if not _async_alloc_works():
        pytest.skip("hipMallocAsync is not functional on this GPU")

    device = torch.device("cuda")
    torch.manual_seed(0)

    x = torch.randn((tokens, HIDDEN_SIZE), dtype=dtype, device=device)
    w_gate_up = (
        torch.randn((2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype, device=device)
        * 0.05
    )
    w_down = (
        torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype, device=device) * 0.05
    )

    with torch.inference_mode(False), torch.no_grad():
        program = PyTorchProgram(
            _StaticMLP(),
            example_input=(x, w_gate_up, w_down),
            options=DoccOptions(
                target=target,
                device_residency=True,
                symbolic_shapes=False,
                rocm_async_alloc=True,
            ),
            name=f"qwen_mlp_graph_capture_{tokens}",
        )
        program.compile()

        # Warm up on a side stream so one-time setup happens outside capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                program(x, w_gate_up, w_down)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = program(x, w_gate_up, w_down)

        # Replay with fresh inputs written in place: the graph reads the
        # addresses it captured.
        for _ in range(3):
            x.copy_(torch.randn((tokens, HIDDEN_SIZE), dtype=dtype, device=device))
            graph.replay()
            torch.cuda.synchronize()

            ref = _StaticMLP()(x.float(), w_gate_up.float(), w_down.float())
            assert captured_out.device.type == "cuda"
            assert _relative_error(captured_out, ref) < RELATIVE_ERROR[dtype]
