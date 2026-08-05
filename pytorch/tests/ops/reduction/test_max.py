import torch
import torch.nn as nn

from tests import check

# --- amax ---


def test_amax_simple(target: str) -> None:
    class AMaxSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amax(input)

    check(AMaxSimpleNet(), torch.randn(4, 4), target=target)


def test_amax_keepdim(target: str) -> None:
    class AMaxKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amax(input, keepdim=True)

    check(AMaxKeepdimNet(), torch.randn(4, 4), target=target)


def test_amax_dim(target: str) -> None:
    class AMaxDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amax(input, dim=1)

    check(AMaxDimNet(), torch.randn(4, 4), target=target)


def test_amax_dim_keepdim(target: str) -> None:
    class AMaxDimKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amax(input, dim=1, keepdim=True)

    check(AMaxDimKeepdimNet(), torch.randn(4, 4), target=target)


# --- any ---


def test_any_simple(target: str) -> None:
    class AnySimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(
        AnySimpleNet(), torch.tensor([[False, True]], dtype=torch.bool), target=target
    )


def test_any_false(target: str) -> None:
    class AnyFalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(
        AnyFalseNet(), torch.tensor([[False, False]], dtype=torch.bool), target=target
    )


def test_any_dim0(target: str) -> None:
    class AnyDim0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input, 0)

    check(AnyDim0Net(), torch.randn(4, 2) < 0, target=target)


def test_any_dim0_keepdim(target: str) -> None:
    class AnyDim0KeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input, 0, keepdim=True)

    check(AnyDim0KeepdimNet(), torch.randn(4, 2) < 0, target=target)


def test_any_dim1(target: str) -> None:
    class AnyDim1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input, 1)

    check(AnyDim1Net(), torch.randn(4, 2) < 0, target=target)


def test_any_dim1_keepdim(target: str) -> None:
    class AnyDim1KeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input, 1, keepdim=True)

    check(AnyDim1KeepdimNet(), torch.randn(4, 2) < 0, target=target)


def test_any_float(target: str) -> None:
    class AnyFloatNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(AnyFloatNet(), torch.randn(2, 3), target=target)


def test_any_float_false(target: str) -> None:
    class AnyFloatFalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(AnyFloatFalseNet(), torch.zeros(2, 3), target=target)


def test_any_uint8(target: str) -> None:
    class AnyUInt8Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(AnyUInt8Net(), torch.tensor([0, 1, 2], dtype=torch.uint8), target=target)


def test_any_uint8_false(target: str) -> None:
    class AnyUInt8FalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.any(input)

    check(AnyUInt8FalseNet(), torch.tensor([0, 1, 2], dtype=torch.uint8), target=target)


# --- max ---


def test_max_simple(target: str) -> None:
    class MaxSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.max(input)

    check(MaxSimpleNet(), torch.randn(1, 3), target=target)
