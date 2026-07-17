import torch
import torch.nn as nn

from tests import check


def test_simple(target: str) -> None:
    class ReductionMeanSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input)

    check(ReductionMeanSimpleNet(), torch.randn(1, 3), target=target)


def test_dtype(target: str) -> None:
    class ReductionMeanDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, dtype=torch.float64)

    check(ReductionMeanDtypeNet(), torch.randn(1, 3), target=target)


def test_dim(target: str) -> None:
    class ReductionMeanDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, 1)

    check(ReductionMeanDimNet(), torch.randn(4, 4), target=target)


def test_tuple_dim(target: str) -> None:
    class ReductionMeanTupleDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, (0, 1))

    check(ReductionMeanTupleDimNet(), torch.randn(4, 4), target=target)


def test_dim_keepdim(target: str) -> None:
    class ReductionMeanDimKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, 1, keepdim=True)

    check(ReductionMeanDimKeepdimNet(), torch.randn(4, 4), target=target)


def test_tuple_dim_keepdim(target: str) -> None:
    class ReductionMeanTupleDimKeepdimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, (0, 1), keepdim=True)

    check(ReductionMeanTupleDimKeepdimNet(), torch.randn(4, 4), target=target)


def test_dim_dtype(target: str) -> None:
    class ReductionMeanDimDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, 1, dtype=torch.float64)

    check(ReductionMeanDimDtypeNet(), torch.randn(4, 4), target=target)


def test_tuple_dim_dtype(target: str) -> None:
    class ReductionMeanTupleDimDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, (0, 1), dtype=torch.float64)

    check(ReductionMeanTupleDimDtypeNet(), torch.randn(4, 4), target=target)


def test_dim_keepdim_dtype(target: str) -> None:
    class ReductionMeanDimKeepdimDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, 1, keepdim=True, dtype=torch.float64)

    check(ReductionMeanDimKeepdimDtypeNet(), torch.randn(4, 4), target=target)


def test_tuple_dim_keepdim_dtype(target: str) -> None:
    class ReductionMeanTupleDimKeepdimDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.mean(input, (0, 1), keepdim=True, dtype=torch.float64)

    check(ReductionMeanTupleDimKeepdimDtypeNet(), torch.randn(4, 4), target=target)
