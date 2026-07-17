import torch
import torch.nn as nn

from tests import check


def test_simple(target: str) -> None:
    class PointwiseAbsSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.abs(input)

    check(PointwiseAbsSimpleNet(), torch.tensor([-1, -2, 3]), target=target)


def test_absolute_simple(target: str) -> None:
    class PointwiseAbsAbsoluteSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.absolute(input)

    check(PointwiseAbsAbsoluteSimpleNet(), torch.tensor([-1, -2, 3]), target=target)
