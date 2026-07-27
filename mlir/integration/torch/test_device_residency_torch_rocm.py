"""Device-resident promotion (rocm)"""

from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import pytest

from docc.torch import compile_torch
from docc.compiler import DoccPerformanceWarning


class _ElementwiseAdd(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _ParamMatmul(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


class _ConvBatchNorm(nn.Module):
    """Conv + BatchNorm: on an accelerator torch routes these through
    cuDNN / MIOpen fused variants that torch-mlir cannot lower. Exercising this
    model validates that vendor backends are disabled during the trace."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


@contextmanager
def _no_perf_warning():
    """Assert that no DoccPerformanceWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DoccPerformanceWarning)
        yield


def _compile_elementwise(target="rocm"):
    """Compile the elementwise model and return (program, x, y)."""
    x = torch.randn(1024)
    y = torch.randn(1024)
    program = compile_torch(_ElementwiseAdd().eval(), (x, y), target=target)
    program.compile()
    return program, x, y


def _compile_matmul(target="rocm"):
    """Compile the gemm model and return (program, x, weight)."""
    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    program = compile_torch(_ParamMatmul(weight).eval(), x, target=target)
    program.compile()
    return program, x, weight


# --------------------------------------------------------------------------- #
# Promotion decision
# --------------------------------------------------------------------------- #


@pytest.mark.rocm()
def test_elementwise_promoted_device_resident():
    program, *_ = _compile_elementwise("rocm")
    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "rocm"


@pytest.mark.rocm()
def test_matmul_promoted_device_resident():
    # The cuBLAS gemm's transfers are extracted into offloading nodes, so the
    # arguments are boundary-only and stay device-resident.
    program, *_ = _compile_matmul("rocm")
    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "rocm"


# --------------------------------------------------------------------------- #
# Device-resident artifact -- call-mode matrix
# --------------------------------------------------------------------------- #


@pytest.mark.rocm()
def test_elementwise_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""
    program, x, y = _compile_elementwise("rocm")
    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda(), y.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.rocm()
def test_elementwise_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""
    program, x, y = _compile_elementwise("rocm")
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x, y)
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.rocm()
def test_matmul_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""
    program, x, weight = _compile_matmul("rocm")
    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)


@pytest.mark.rocm()
def test_matmul_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""
    program, x, weight = _compile_matmul("rocm")
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x)
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------- #
# Vendor-specific op backends disabled during tracing
#
# torch-mlir cannot lower vendor-fused ops (conv/batchnorm routed through
# cuDNN / MIOpen). TorchProgram disables those backends for the duration of the
# trace. We assert the *mechanism* (flag state during/after the trace) rather
# than that a particular op is unsupported, since torch-mlir op coverage changes
# over time and a "this op fails" test would silently rot.
# --------------------------------------------------------------------------- #


def test_export_backend_flags_toggle_and_restore():
    """The context manager disables cuDNN during the block and restores the
    prior state afterwards, without leaking global state."""
    from docc.torch.torch_program import _export_backend_flags

    before = torch.backends.cudnn.enabled
    with _export_backend_flags():
        assert torch.backends.cudnn.enabled is False
    assert torch.backends.cudnn.enabled == before


def test_vendor_backends_disabled_during_trace():
    """The vendor backends must be disabled *while the model is traced*, so
    torch-mlir never sees a cuDNN / MIOpen-fused op. We record the flag state
    from inside forward (which runs during export) instead of depending on a
    specific op being unsupported."""
    pytest.importorskip("torch_mlir")

    seen = {}

    class _RecordCudnnState(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen["cudnn_enabled"] = torch.backends.cudnn.enabled
            return x + 1.0

    before = torch.backends.cudnn.enabled
    x = torch.randn(8)
    program = compile_torch(_RecordCudnnState().eval(), x, target="none")
    program.to_sdfg()

    # cuDNN was off while forward executed under the tracer ...
    assert seen["cudnn_enabled"] is False
    # ... and the global state was restored once tracing finished.
    assert torch.backends.cudnn.enabled == before


@pytest.mark.rocm()
def test_conv_batchnorm_compiles_and_runs():
    """End-to-end, on-device trace: a Conv+BatchNorm model whose parameters and
    example input live on the GPU. Tracing in this configuration is exactly what
    makes torch emit MIOpen / cuDNN-fused ops -- which torch-mlir cannot lower --
    so this exercises the vendor-backend disable during export. Without the fix
    the compile would fail at the torch-mlir import step."""
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 3, 8, 8)
    model = _ConvBatchNorm().eval()

    with torch.no_grad():
        expected = model(x_cpu)

    # Move model and the example input onto the device so the trace runs on the
    # GPU (the fused-op scenario).
    model_cuda = model.cuda()
    x_cuda = x_cpu.cuda()

    program = compile_torch(model_cuda, x_cuda, target="rocm")
    program.compile()
    with torch.no_grad():
        res = program(x_cuda)

    torch.testing.assert_close(res.cpu(), expected, rtol=1e-3, atol=1e-4)


@pytest.mark.rocm()
def test_conv_batchnorm_repeated_calls_stable():
    """Repeated calls on the on-device workflow must not re-trace the model
    (which would re-enter the vendor-fused ops); the cached artifact keeps
    returning correct results."""
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 3, 8, 8)
    model = _ConvBatchNorm().eval()

    with torch.no_grad():
        expected = model(x_cpu)

    model_cuda = model.cuda()
    x_cuda = x_cpu.cuda()

    program = compile_torch(model_cuda, x_cuda, target="rocm")
    program.compile()

    with torch.no_grad():
        first = program(x_cuda)
        second = program(x_cuda)

    torch.testing.assert_close(first.cpu(), expected, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(second.cpu(), expected, rtol=1e-3, atol=1e-4)
