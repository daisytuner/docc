import torch
import torch.nn as nn

from tests import check


def test_tensor_tensor_float(target: str) -> None:
    class EqTensorNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.eq(x, y)

    x = torch.tensor([[1.0, 2.0, 3.2], [3.0, 4.0, 3.2]])
    y = torch.tensor([[1.0, 5.0, 3.2], [3.0, 6.0, 3.2]])
    check(EqTensorNet(), x, y, target=target)


def test_tensor_tensor_int(target: str) -> None:
    class EqTensorNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.eq(x, y)

    x = torch.tensor([[1, 2, 3], [3, 4, 5]])
    y = torch.tensor([[1, 5, 3], [3, 6, 5]])
    check(EqTensorNet(), x, y, target=target)


def test_tensor_tensor_same_container(target: str) -> None:
    class EqSameTensorNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.eq(x, x)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    check(EqSameTensorNet(), x, target=target)


def test_tensor_scalar_float(target: str) -> None:
    class EqScalarFloatNet(nn.Module):
        def forward(self, x: torch.Tensor, y: float) -> torch.Tensor:
            return torch.eq(x, y)

    x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 2.0, 4.0]])
    check(EqScalarFloatNet(), x, 2.0, target=target)


def test_tensor_scalar_int(target: str) -> None:
    class EqScalarIntNet(nn.Module):
        def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
            return torch.eq(x, y)

    x = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    check(EqScalarIntNet(), x, 2, target=target)


def test_tensor_tensor_bool(target: str) -> None:
    torch._dynamo.reset()

    class EqTensorBoolNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.eq(x, y)

    x = torch.tensor([True, False, True])
    y = torch.tensor([True, True, False])
    check(EqTensorBoolNet(), x, y, target=target)
