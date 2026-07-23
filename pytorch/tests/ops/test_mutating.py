import torch
import torch.nn as nn

from tests import check

# --- permute ---


def test_permute_simple(target: str) -> None:
    class PermuteSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.permute(input, (2, 0, 1))

    check(PermuteSimpleNet(), torch.randn(2, 3, 5), target=target)


# --- squeeze ---


def test_squeeze_simple(target: str) -> None:
    class SqueezeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input)

    check(SqueezeSimpleNet(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_0(target: str) -> None:
    class Squeeze0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, 0)

    check(Squeeze0Net(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_1(target: str) -> None:
    class Squeeze1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, 1)

    check(Squeeze1Net(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_tuple(target: str) -> None:
    class SqueezeTupleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, (1, 2, 3))

    check(SqueezeTupleNet(), torch.randn(2, 1, 2, 1, 2), target=target)


# --- unsqueeze ---


def test_unsqueeze_0(target: str) -> None:
    class Unsqueeze0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(input, 0)

    check(Unsqueeze0Net(), torch.tensor([1, 2, 3, 4]), target=target)


def test_unsqueeze_1(target: str) -> None:
    class Unsqueeze1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(input, 1)

    check(Unsqueeze1Net(), torch.tensor([1, 2, 3, 4]), target=target)
