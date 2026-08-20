"""Device-resident promotion (cuda)"""

from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch._dynamo
import pytest

from docc.torch import compile_torch
from docc.compiler import DoccPerformanceWarning


@contextmanager
def _no_perf_warning():
    """Assert that no DoccPerformanceWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DoccPerformanceWarning)
        yield


# --------------------------------------------------------------------------- #
# Promotion decision
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_elementwise_promoted_device_resident():
    class _ElementwiseAdd1(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    program = compile_torch(_ElementwiseAdd1().eval(), (x, y), target="cuda")
    program.compile()

    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "cuda"


@pytest.mark.cuda()
def test_matmul_promoted_device_resident():
    # The cuBLAS gemm's transfers are extracted into offloading nodes, so the
    # arguments are boundary-only and stay device-resident.
    class _ParamMatmul1(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, self.weight)

    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    program = compile_torch(_ParamMatmul1(weight).eval(), x, target="cuda")
    program.compile()

    assert program._compiled.device_resident is True
    assert program._compiled.device_backend == "cuda"


# --------------------------------------------------------------------------- #
# Device-resident artifact -- call-mode matrix
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_elementwise_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""

    class _ElementwiseAdd2(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    program = compile_torch(_ElementwiseAdd2().eval(), (x, y), target="cuda")
    program.compile()

    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda(), y.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda()
def test_elementwise_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""

    class _ElementwiseAdd3(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    program = compile_torch(_ElementwiseAdd3().eval(), (x, y), target="cuda")
    program.compile()

    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x, y)
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda()
def test_matmul_with_cuda_tensors():
    """CUDA tensors pass straight through (zero-copy), no warning, device out."""

    class _ParamMatmul2(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, self.weight)

    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    program = compile_torch(_ParamMatmul2(weight).eval(), x, target="cuda")
    program.compile()

    with _no_perf_warning(), torch.no_grad():
        res = program(x.cuda())
    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda()
def test_matmul_with_cpu_tensors_warns():
    """CPU tensors are copied to the device with a host-to-device perf warning."""

    class _ParamMatmul3(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, self.weight)

    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    program = compile_torch(_ParamMatmul3(weight).eval(), x, target="cuda")
    program.compile()

    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = program(x)
    torch.testing.assert_close(res.cpu(), x @ weight, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------- #
# torch.compile + on_compile callback
# --------------------------------------------------------------------------- #
#
# Via torch.compile the artifact is built eagerly when the backend runs (on the
# first call), so residency is reported through the on_compile callback rather
# than read off a program handle. The callback fires once per compiled graph
# ("inference" for the no-grad dynamo path).


@pytest.mark.cuda()
def test_compile_elementwise_promoted_device_resident():
    """on_compile reports a device-resident inference artifact for x + y."""

    class _ElementwiseAddC1(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    events: list = []
    torch._dynamo.reset()
    compiled = torch.compile(
        _ElementwiseAddC1().eval(),
        backend="docc",
        options={"target": "cuda", "on_compile": events.append},
    )
    with _no_perf_warning(), torch.no_grad():
        compiled(x.cuda(), y.cuda())

    assert len(events) == 1
    info = events[0]
    assert info["graph"] == "inference"
    assert info["device_resident"] is True
    assert info["device_backend"] == "cuda"
    assert info["sdfg"] is not None


@pytest.mark.cuda()
def test_compile_matmul_promoted_device_resident():
    # The cuBLAS gemm's transfers are extracted into offloading nodes, so the
    # arguments are boundary-only and stay device-resident.
    class _ParamMatmulC1(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, self.weight)

    x = torch.randn(32, 32)
    weight = torch.randn(32, 16)
    events: list = []
    torch._dynamo.reset()
    compiled = torch.compile(
        _ParamMatmulC1(weight).cuda().eval(),
        backend="docc",
        options={"target": "cuda", "on_compile": events.append},
    )
    with _no_perf_warning(), torch.no_grad():
        compiled(x.cuda())

    assert len(events) == 1
    info = events[0]
    assert info["graph"] == "inference"
    assert info["device_resident"] is True
    assert info["device_backend"] == "cuda"
    assert info["sdfg"] is not None


@pytest.mark.cuda()
def test_compile_elementwise_with_cuda_tensors():
    """CUDA tensors pass zero-copy through a torch.compile'd device-resident model."""

    class _ElementwiseAddC2(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    events: list = []
    torch._dynamo.reset()
    compiled = torch.compile(
        _ElementwiseAddC2().eval(),
        backend="docc",
        options={"target": "cuda", "on_compile": events.append},
    )
    with _no_perf_warning(), torch.no_grad():
        res = compiled(x.cuda(), y.cuda())

    assert res.is_cuda
    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)
    assert events[-1]["device_resident"] is True


@pytest.mark.cuda()
def test_compile_elementwise_with_cpu_tensors_warns():
    """CPU tensors into a device-resident torch.compile'd model warn once."""

    class _ElementwiseAddC3(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    x = torch.randn(1024)
    y = torch.randn(1024)
    events: list = []
    torch._dynamo.reset()
    compiled = torch.compile(
        _ElementwiseAddC3().eval(),
        backend="docc",
        options={"target": "cuda", "on_compile": events.append},
    )
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        with torch.no_grad():
            res = compiled(x, y)

    torch.testing.assert_close(res.cpu(), x + y, rtol=1e-5, atol=1e-5)
    assert events[-1]["device_resident"] is True
