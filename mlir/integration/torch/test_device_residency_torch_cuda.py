"""Device-resident promotion (cuda)"""

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


@contextmanager
def _no_perf_warning():
    """Assert that no DoccPerformanceWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DoccPerformanceWarning)
        yield


def _compile_elementwise(target="cuda"):
    """Compile the elementwise model and return (program, x, y)."""
    x = torch.randn(1024)
    y = torch.randn(1024)
    program = compile_torch(_ElementwiseAdd().eval(), (x, y), target=target)
    program.compile()
    return program, x, y


def _compile_matmul(target="cuda"):
    """Compile the gemm model and return (program, x, weight)."""
    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    program = compile_torch(_ParamMatmul(weight).eval(), x, target=target)
    program.compile()
    return program, x, weight


# --------------------------------------------------------------------------- #
# Promotion decision
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_elementwise_promoted_device_resident():
    program, *_ = _compile_elementwise("cuda")
    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "cuda"


@pytest.mark.cuda()
def test_matmul_promoted_device_resident():
    # The cuBLAS gemm's transfers are extracted into offloading nodes, so the
    # arguments are boundary-only and stay device-resident.
    program, *_ = _compile_matmul("cuda")
    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "cuda"


# --------------------------------------------------------------------------- #
# Device-resident artifact -- call-mode matrix
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_elementwise_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""
    program, x, y = _compile_elementwise("cuda")
    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda(), y.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda()
def test_elementwise_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""
    program, x, y = _compile_elementwise("cuda")
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x, y)
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda()
def test_matmul_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""
    program, x, weight = _compile_matmul("cuda")
    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda()
def test_matmul_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""
    program, x, weight = _compile_matmul("cuda")
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x)
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)
