import torch
import torch.nn as nn

from tests import check


def test_acos_simple(target: str) -> None:
    class PointwiseCMathAcosSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.acos(input)

    check(PointwiseCMathAcosSimpleNet(), torch.randn(4), target=target, equal_nan=True)


def test_arccos_simple(target: str) -> None:
    class PointwiseCMathArccosSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arccos(input)

    check(
        PointwiseCMathArccosSimpleNet(), torch.randn(4), target=target, equal_nan=True
    )


def test_acosh_simple(target: str) -> None:
    class PointwiseCMathAcoshSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.acosh(input)

    check(
        PointwiseCMathAcoshSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_arccosh_simple(target: str) -> None:
    class PointwiseCMathArccoshSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arccosh(input)

    check(
        PointwiseCMathArccoshSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_asin_simple(target: str) -> None:
    class PointwiseCMathAsinSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.asin(input)

    check(PointwiseCMathAsinSimpleNet(), torch.randn(4), target=target, equal_nan=True)


def test_arcsin_simple(target: str) -> None:
    class PointwiseCMathArcsinSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arcsin(input)

    check(
        PointwiseCMathArcsinSimpleNet(), torch.randn(4), target=target, equal_nan=True
    )


def test_asinh_simple(target: str) -> None:
    class PointwiseCMathAsinhSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.asinh(input)

    check(
        PointwiseCMathAsinhSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_arcsinh_simple(target: str) -> None:
    class PointwiseCMathArcsinhSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arcsinh(input)

    check(
        PointwiseCMathArcsinhSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_atan_simple(target: str) -> None:
    class PointwiseCMathAtanSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.atan(input)

    check(PointwiseCMathAtanSimpleNet(), torch.randn(4), target=target, equal_nan=True)


def test_arctan_simple(target: str) -> None:
    class PointwiseCMathArctanSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arctan(input)

    check(
        PointwiseCMathArctanSimpleNet(), torch.randn(4), target=target, equal_nan=True
    )


def test_atanh_simple(target: str) -> None:
    class PointwiseCMathAtanhSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.atanh(input)

    check(
        PointwiseCMathAtanhSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_arctanh_simple(target: str) -> None:
    class PointwiseCMathArctanhSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.arctanh(input)

    check(
        PointwiseCMathArctanhSimpleNet(),
        torch.randn(4).uniform_(1, 2),
        target=target,
        equal_nan=True,
    )


def test_atan2_simple(target: str) -> None:
    class PointwiseCMathAtan2SimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.atan2(input, other)

    check(
        PointwiseCMathAtan2SimpleNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target,
        equal_nan=True
    )


def test_arctan2_simple(target: str) -> None:
    class PointwiseCMathArctan2SimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.arctan2(input, other)

    check(
        PointwiseCMathArctan2SimpleNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target,
        equal_nan=True
    )
