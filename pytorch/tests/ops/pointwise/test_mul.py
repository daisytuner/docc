import torch
import torch.nn as nn

from tests import check

# --- mul ---


def test_mul_tensor_tensor(target: str) -> None:
    class MulTensorNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.mul(x, y)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    check(MulTensorNet(), x, y, target=target)


def test_mul_tensor_scalar_float(target: str) -> None:
    class MulScalarFloatNet(nn.Module):
        def forward(self, x: torch.Tensor, y: float) -> torch.Tensor:
            return torch.mul(x, y)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    check(MulScalarFloatNet(), x, 2.5, target=target)


def test_mul_tensor_scalar_int(target: str) -> None:
    class MulScalarIntNet(nn.Module):
        def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
            return torch.mul(x, y)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    check(MulScalarIntNet(), x, 3, target=target)


def test_mul_broadcast(target: str) -> None:
    class PointwiseMulBroadcastNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.mul(input, other)

    check(
        PointwiseMulBroadcastNet(), *(torch.randn(3), torch.randn(2, 3)), target=target
    )
