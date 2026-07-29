import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check


def test_functional_simple(target: str) -> None:
    class PointwiseSigmoidFunctionalSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.sigmoid(input)

    check(PointwiseSigmoidFunctionalSimpleNet(), torch.randn(4), target=target)


def test_expit_simple(target: str) -> None:
    class PointwiseSigmoidExpitSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.special.expit(input)

    check(PointwiseSigmoidExpitSimpleNet(), torch.randn(4), target=target)


def test_float32(target: str) -> None:
    class PointwiseSigmoidSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(input)

    check(
        PointwiseSigmoidSimpleNet(),
        torch.randn(4, dtype=torch.float32),
        target=target,
    )


def test_float64(target: str) -> None:
    class PointwiseSigmoidSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(input)

    check(
        PointwiseSigmoidSimpleNet(),
        torch.randn(4, dtype=torch.float64),
        target=target,
    )
