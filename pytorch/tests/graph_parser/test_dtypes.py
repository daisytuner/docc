"""
Test GraphParser on all supported PyTorch dtypes.
"""

import torch
import torch.nn as nn

import pytest

from tests import check


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float, id="float"),
        pytest.param(torch.float64, id="float64"),
        pytest.param(torch.double, id="double"),
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.half, id="half"),
        pytest.param(torch.bfloat16, id="bfloat16"),
    ],
)
def test_dtype_floating_point(target: str, dtype: torch.dtype) -> None:
    class DTypeFloatingPointNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(5, dtype=dtype)
    y = torch.randn(5, dtype=dtype)
    check(DTypeFloatingPointNet(), *(x, y), target=target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.uint8, id="uint8"),
        pytest.param(torch.uint16, id="uint16"),
        pytest.param(torch.uint32, id="uint32"),
        pytest.param(torch.uint64, id="uint64"),
    ],
)
def test_dtype_unsigned_integer(target: str, dtype: torch.dtype) -> None:
    class DTypeUnsignedIntegerNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.mul(x, y)

    x = torch.randint(0, 10, (5,), dtype=dtype)
    y = torch.randint(0, 10, (5,), dtype=dtype)
    check(DTypeUnsignedIntegerNet(), *(x, y), target=target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int8, id="int8"),
        pytest.param(torch.int16, id="int16"),
        pytest.param(torch.short, id="short"),
        pytest.param(torch.int32, id="int32"),
        pytest.param(torch.int, id="int"),
        pytest.param(torch.int64, id="int64"),
        pytest.param(torch.long, id="long"),
    ],
)
def test_dtype_signed_integer(target: str, dtype: torch.dtype) -> None:
    class DTypeSignedIntegerNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 10, (5,), dtype=dtype)
    y = torch.randint(0, 10, (5,), dtype=dtype)
    check(DTypeSignedIntegerNet(), *(x, y), target=target)


def test_dtype_bool(target: str):
    class DTypeBoolNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.bitwise_and(x, y)

    x = torch.randint(0, 1, (5,), dtype=torch.bool)
    y = torch.randint(0, 1, (5,), dtype=torch.bool)
    check(DTypeBoolNet(), *(x, y), target=target)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(torch.add, id="add"),
        pytest.param(torch.sub, id="sub"),
        pytest.param(torch.mul, id="mul"),
        pytest.param(torch.div, id="div"),
        pytest.param(torch.atan2, id="atan2"),
        pytest.param(torch.eq, id="eq"),
        pytest.param(torch.pow, id="pow"),
    ],
)
@pytest.mark.parametrize(
    "x_dtype, y_dtype",
    [
        pytest.param(torch.bfloat16, torch.float32, id="bfloat16-float32"),
        pytest.param(torch.float32, torch.float16, id="float32-float16"),
        pytest.param(torch.bfloat16, torch.float16, id="bfloat16-float16"),
        pytest.param(torch.float16, torch.float64, id="float16-float64"),
    ],
)
def test_dtype_mixed(
    target: str, op, x_dtype: torch.dtype, y_dtype: torch.dtype
) -> None:
    """Operands of different types compute in their promoted type, as in PyTorch."""

    class DTypeMixedNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return op(x, y)

    x = torch.rand(5, dtype=x_dtype) + 0.5
    y = torch.rand(5, dtype=y_dtype) + 0.5
    check(DTypeMixedNet(), *(x, y), rtol=1e-2, atol=1e-2, target=target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float8_e4m3fn, id="float8_e4m3fn"),
        pytest.param(torch.float8_e5m2, id="float8_e5m2"),
    ],
)
def test_dtype_float8(target: str, dtype: torch.dtype) -> None:
    class DTypeFloat8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float32) * y

    x = torch.randn(5).to(dtype)
    y = torch.randn(5)
    check(DTypeFloat8Net(), *(x, y), target=target)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
    ],
)
def test_dtype_rms_norm(target: str, dtype: torch.dtype) -> None:
    """Normalizes in float32 and scales in the input's type, as Hugging Face's RMSNorm does."""

    class DTypeRMSNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.rand(16, dtype=dtype) + 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x.to(torch.float32)
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
            return self.weight * h.to(x.dtype)

    x = torch.randn(4, 16, dtype=dtype)
    check(DTypeRMSNormNet(), x, rtol=1e-2, atol=1e-2, target=target)
