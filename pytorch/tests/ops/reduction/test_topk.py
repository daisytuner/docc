import torch
import torch.nn as nn

from tests import check


def test_last_dim(target: str) -> None:
    class TopKLastDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 3)
            return values, indices

    check(TopKLastDimNet(), torch.randn(4, 16), target=target)


def test_dim_0(target: str) -> None:
    class TopKDim0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 2, dim=0)
            return values, indices

    check(TopKDim0Net(), torch.randn(8, 5), target=target)


def test_middle_dim(target: str) -> None:
    class TopKMiddleDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 2, dim=1)
            return values, indices

    check(TopKMiddleDimNet(), torch.randn(2, 6, 3), target=target)


def test_vector(target: str) -> None:
    class TopKVectorNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 4)
            return values, indices

    check(TopKVectorNet(), torch.randn(32), target=target)


def test_smallest(target: str) -> None:
    class TopKSmallestNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 3, largest=False)
            return values, indices

    check(TopKSmallestNet(), torch.randn(4, 16), target=target)


def test_k_1(target: str) -> None:
    class TopKOneNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 1)
            return values, indices

    check(TopKOneNet(), torch.randn(4, 16), target=target)


def test_integers(target: str) -> None:
    class TopKIntegerNet(nn.Module):
        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            values, indices = torch.topk(input, 3)
            return values, indices

    check(
        TopKIntegerNet(),
        torch.randperm(16, dtype=torch.int32).reshape(2, 8),
        target=target,
    )


def test_renormalized_router(target: str) -> None:
    """The routing of a mixture-of-experts layer: the top-k of the scores' softmax, renormalized."""

    class TopKRouterNet(nn.Module):
        def forward(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weights, experts = torch.topk(torch.softmax(scores, dim=-1), 2, dim=-1)
            return weights / weights.sum(dim=-1, keepdim=True), experts

    check(TopKRouterNet(), torch.randn(3, 8), target=target)
