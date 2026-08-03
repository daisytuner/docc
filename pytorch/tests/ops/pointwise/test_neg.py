import torch
import torch.nn as nn

from tests import check


def test_simple(target: str) -> None:
    class PointwiseNegSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.neg(input)

    check(PointwiseNegSimpleNet(), *(torch.randn(4),), target=target)


def test_multidim(target: str) -> None:
    class PointwiseNegMultidimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.neg(input)

    check(PointwiseNegMultidimNet(), *(torch.randn(2, 3, 4),), target=target)


def test_integer(target: str) -> None:
    class PointwiseNegIntegerNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.neg(input)

    check(PointwiseNegIntegerNet(), torch.tensor([-1, -2, 3]), target=target)


def test_negative_simple(target: str) -> None:
    class PointwiseNegNegativeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.negative(input)

    check(PointwiseNegNegativeSimpleNet(), *(torch.randn(4),), target=target)
