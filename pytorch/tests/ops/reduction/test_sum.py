import torch
import torch.nn as nn

from tests import check


def test_simple(target: str) -> None:
    class ReductionSumSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input)

    check(ReductionSumSimpleNet(), torch.randn(1, 3), target=target)


def test_dtype(target: str) -> None:
    class ReductionSumDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input, dtype=torch.float64)

    check(ReductionSumDtypeNet(), torch.randn(1, 3), target=target)


def test_dim(target: str) -> None:
    class ReductionSumDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input, 1)

    check(ReductionSumDimNet(), torch.randn(4, 4), target=target)


def test_tuple_dim(target: str) -> None:
    class ReductionSumTupleDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input, (0, 1))

    check(ReductionSumTupleDimNet(), torch.randn(4, 4), target=target)


def test_dim_keepdim(target: str) -> None:
    class ReductionSumDimKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.sum(dim=-1, keepdim=True)

    check(ReductionSumDimKeepdimNet(), torch.randn(4, 4), target=target)


def test_dim_none(target: str) -> None:
    class ReductionSumDimNoneNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input, dim=None, keepdim=True)

    check(ReductionSumDimNoneNet(), torch.randn(4, 4), target=target)


def test_dim_dtype(target: str) -> None:
    class ReductionSumDimDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.sum(input, 1, dtype=torch.float64)

    check(ReductionSumDimDtypeNet(), torch.randn(4, 4), target=target)
