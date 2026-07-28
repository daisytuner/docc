import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check


def test_method_simple(target: str) -> None:
    class PointwiseSigmoidMethodSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.sigmoid()

    check(PointwiseSigmoidMethodSimpleNet(), torch.randn(4), target=target)


def test_functional_simple(target: str) -> None:
    class PointwiseSigmoidFunctionalSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.sigmoid(input)

    check(PointwiseSigmoidFunctionalSimpleNet(), torch.randn(4), target=target)


def test_module_simple(target: str) -> None:
    class PointwiseSigmoidModuleSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sigmoid = nn.Sigmoid()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.sigmoid(input)

    check(PointwiseSigmoidModuleSimpleNet(), torch.randn(4), target=target)


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
