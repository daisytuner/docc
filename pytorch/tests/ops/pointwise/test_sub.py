import torch
import torch.nn as nn

from tests import check

# --- sub ---


def test_sub_simple(target: str) -> None:
    class PointwiseSubSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.sub(input, other)

    check(PointwiseSubSimpleNet(), *(torch.randn(4), torch.randn(4)), target=target)


def test_sub_constant_float_alpha(target: str) -> None:
    class PointwiseSubConstantFloatAlphaNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.sub(input, other, alpha=10.0)

    check(
        PointwiseSubConstantFloatAlphaNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target
    )


def test_sub_constant_int_alpha(target: str) -> None:
    class PointwiseSubConstantIntAlphaNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.sub(input, other, alpha=10)

    check(
        PointwiseSubConstantIntAlphaNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target
    )


def test_sub_float_alpha(target: str) -> None:
    class PointwiseSubFloatAlphaNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return torch.sub(input, other, alpha=alpha)

    check(
        PointwiseSubFloatAlphaNet(),
        *(torch.randn(4), torch.randn(4), 10.0),
        target=target
    )


def test_sub_int_alpha(target: str) -> None:
    class PointwiseSubIntAlphaNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: torch.Tensor, alpha: int
        ) -> torch.Tensor:
            return torch.sub(input, other, alpha=alpha)

    check(
        PointwiseSubIntAlphaNet(), *(torch.randn(4), torch.randn(4), 10), target=target
    )


def test_sub_scalar(target: str) -> None:
    class PointwiseSubSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: float) -> torch.Tensor:
            return torch.sub(input, other)

    check(PointwiseSubSimpleNet(), *(torch.randn(4), 0.2), target=target)


def test_sub_scalar_float_alpha(target: str) -> None:
    class PointwiseSubSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: float, alpha: float
        ) -> torch.Tensor:
            return torch.sub(input, other, alpha=alpha)

    check(PointwiseSubSimpleNet(), *(torch.randn(4), 0.2, 10.0), target=target)


def test_sub_scalar_int_alpha(target: str) -> None:
    class PointwiseSubSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: float, alpha: int
        ) -> torch.Tensor:
            return torch.sub(input, other, alpha=alpha)

    check(PointwiseSubSimpleNet(), *(torch.randn(4), 0.2, 10), target=target)


def test_sub_broadcast(target: str) -> None:
    class PointwiseSubBroadcastNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.sub(input, other)

    check(
        PointwiseSubBroadcastNet(),
        *(torch.randn(8, 16, 32), torch.randn(32)),
        target=target
    )
