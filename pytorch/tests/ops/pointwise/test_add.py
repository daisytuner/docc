import torch
import torch.nn as nn

from tests import check


def test_simple(target: str) -> None:
    class PointwiseAddSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.add(input, other)

    check(PointwiseAddSimpleNet(), *(torch.randn(4), torch.randn(4)), target=target)


def test_constant_float_alpha(target: str) -> None:
    class PointwiseAddConstantFloatAlphaNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.add(input, other, alpha=10.0)

    check(
        PointwiseAddConstantFloatAlphaNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target
    )


def test_constant_int_alpha(target: str) -> None:
    class PointwiseAddConstantIntAlphaNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.add(input, other, alpha=10)

    check(
        PointwiseAddConstantIntAlphaNet(),
        *(torch.randn(4), torch.randn(4)),
        target=target
    )


def test_float_alpha(target: str) -> None:
    class PointwiseAddFloatAlphaNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: torch.Tensor, alpha: float
        ) -> torch.Tensor:
            return torch.add(input, other, alpha=alpha)

    check(
        PointwiseAddFloatAlphaNet(),
        *(torch.randn(4), torch.randn(4), 10.0),
        target=target
    )


def test_int_alpha(target: str) -> None:
    class PointwiseAddIntAlphaNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: torch.Tensor, alpha: int
        ) -> torch.Tensor:
            return torch.add(input, other, alpha=alpha)

    check(
        PointwiseAddIntAlphaNet(), *(torch.randn(4), torch.randn(4), 10), target=target
    )


def test_scalar(target: str) -> None:
    class PointwiseAddSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: float) -> torch.Tensor:
            return torch.add(input, other)

    check(PointwiseAddSimpleNet(), *(torch.randn(4), 0.2), target=target)


def test_scalar_float_alpha(target: str) -> None:
    class PointwiseAddSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: float, alpha: float
        ) -> torch.Tensor:
            return torch.add(input, other, alpha=alpha)

    check(PointwiseAddSimpleNet(), *(torch.randn(4), 0.2, 10.0), target=target)


def test_scalar_int_alpha(target: str) -> None:
    class PointwiseAddSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, other: float, alpha: int
        ) -> torch.Tensor:
            return torch.add(input, other, alpha=alpha)

    check(PointwiseAddSimpleNet(), *(torch.randn(4), 0.2, 10), target=target)


def test_addcdiv_simple(target: str) -> None:
    class PointwiseAddcdivSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addcdiv(input, tensor1, tensor2)

    check(
        PointwiseAddcdivSimpleNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3)),
        target=target
    )


def test_addcdiv_float_value(target: str) -> None:
    class PointwiseAddcdivFloatValueNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            tensor1: torch.Tensor,
            tensor2: torch.Tensor,
            value: float,
        ) -> torch.Tensor:
            return torch.addcdiv(input, tensor1, tensor2, value=value)

    check(
        PointwiseAddcdivFloatValueNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3), 0.1),
        target=target
    )


def test_addcdiv_int_value(target: str) -> None:
    class PointwiseAddcdivFloatValueNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            tensor1: torch.Tensor,
            tensor2: torch.Tensor,
            value: int,
        ) -> torch.Tensor:
            return torch.addcdiv(input, tensor1, tensor2, value=value)

    check(
        PointwiseAddcdivFloatValueNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3), 2),
        target=target
    )


def test_addcmul_simple(target: str) -> None:
    class PointwiseAddcmulSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addcmul(input, tensor1, tensor2)

    check(
        PointwiseAddcmulSimpleNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3)),
        target=target
    )


def test_addcmul_float_value(target: str) -> None:
    class PointwiseAddcmulFloatValueNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            tensor1: torch.Tensor,
            tensor2: torch.Tensor,
            value: float,
        ) -> torch.Tensor:
            return torch.addcmul(input, tensor1, tensor2, value=value)

    check(
        PointwiseAddcmulFloatValueNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3), 0.1),
        target=target
    )


def test_addcmul_int_value(target: str) -> None:
    class PointwiseAddcmulFloatValueNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            tensor1: torch.Tensor,
            tensor2: torch.Tensor,
            value: int,
        ) -> torch.Tensor:
            return torch.addcmul(input, tensor1, tensor2, value=value)

    check(
        PointwiseAddcmulFloatValueNet(),
        *(torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3), 2),
        target=target
    )
