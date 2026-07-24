import torch
import torch.nn as nn

from tests import check

# --- full ---


def test_full_simple(target: str) -> None:
    class FullSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.full((2, 3), 3.141592)

    check(FullSimpleNet(), *(), target=target)


def test_full_dtype(target: str) -> None:
    class FullDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.full((2, 3), 3.141592, dtype=torch.float64)

    check(FullDtypeNet(), *(), target=target)


# --- full_like ---


def test_full_like_simple(target: str) -> None:
    class FullLikeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592)

    check(FullLikeSimpleNet(), torch.ones(2, 3), target=target)


def test_full_like_dtype(target: str) -> None:
    class FullLikeDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592)

    check(FullLikeDtypeNet(), torch.ones((2, 3), dtype=torch.float64), target=target)


def test_full_like_dtype_change(target: str) -> None:
    class FullLikeDtypeChangeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592, dtype=torch.float32)

    check(
        FullLikeDtypeChangeNet(), torch.ones((2, 3), dtype=torch.float64), target=target
    )
