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
