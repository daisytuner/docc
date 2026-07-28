import torch
import torch.nn as nn

from tests import check


def test_logical_not_bool(target: str) -> None:
    class PointwiseLogicalNotSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.logical_not(input)

    check(
        PointwiseLogicalNotSimpleNet(), torch.tensor([True, False, True]), target=target
    )


def test_logical_not_int(target: str) -> None:
    class PointwiseLogicalNotSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.logical_not(input)

    check(PointwiseLogicalNotSimpleNet(), torch.tensor([1, 0, 1]), target=target)


def test_logical_not_float(target: str) -> None:
    class PointwiseLogicalNotSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.logical_not(input)

    check(PointwiseLogicalNotSimpleNet(), torch.tensor([1.0, 0.0, 1.0]), target=target)
