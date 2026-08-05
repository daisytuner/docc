import torch
import torch.nn as nn

from tests import check

# --- amin ---


def test_amin_simple(target: str) -> None:
    class AMinSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amin(input)

    check(AMinSimpleNet(), torch.randn(4, 4), target=target)


def test_amin_keepdim(target: str) -> None:
    class AMinKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amin(input, keepdim=True)

    check(AMinKeepdimNet(), torch.randn(4, 4), target=target)


def test_amin_dim(target: str) -> None:
    class AMinDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amin(input, dim=1)

    check(AMinDimNet(), torch.randn(4, 4), target=target)


def test_amin_dim_keepdim(target: str) -> None:
    class AMinDimKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.amin(input, dim=1, keepdim=True)

    check(AMinDimKeepdimNet(), torch.randn(4, 4), target=target)


# --- all ---


def test_all_simple(target: str) -> None:
    class AllSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(
        AllSimpleNet(), torch.tensor([[False, True]], dtype=torch.bool), target=target
    )


def test_all_false(target: str) -> None:
    class AllFalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(
        AllFalseNet(), torch.tensor([[False, False]], dtype=torch.bool), target=target
    )


def test_all_dim0(target: str) -> None:
    class AllDim0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input, 0)

    check(AllDim0Net(), torch.randn(4, 2) < 0, target=target)


def test_all_dim0_keepdim(target: str) -> None:
    class AllDim0KeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input, 0, keepdim=True)

    check(AllDim0KeepdimNet(), torch.randn(4, 2) < 0, target=target)


def test_all_dim1(target: str) -> None:
    class AllDim1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input, 1)

    check(AllDim1Net(), torch.randn(4, 2) < 0, target=target)


def test_all_dim1_keepdim(target: str) -> None:
    class AllDim1KeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input, 1, keepdim=True)

    check(AllDim1KeepdimNet(), torch.randn(4, 2) < 0, target=target)


def test_all_float(target: str) -> None:
    class AllFloatNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(AllFloatNet(), torch.randn(2, 3), target=target)


def test_all_float_false(target: str) -> None:
    class AllFloatFalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(AllFloatFalseNet(), torch.zeros(2, 3), target=target)


def test_all_uint8(target: str) -> None:
    class AllUInt8Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(AllUInt8Net(), torch.tensor([0, 1, 2], dtype=torch.uint8), target=target)


def test_all_uint8_false(target: str) -> None:
    class AllUInt8FalseNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.all(input)

    check(AllUInt8FalseNet(), torch.tensor([0, 1, 2], dtype=torch.uint8), target=target)


# --- min ---


def test_min_simple(target: str) -> None:
    class MinSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.min(input)

    check(MinSimpleNet(), torch.randn(1, 3), target=target)
