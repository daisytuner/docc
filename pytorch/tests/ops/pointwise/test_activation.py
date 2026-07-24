import torch
import torch.nn as nn

from tests import check

# --- softmax ---


def test_softmax_simple(target: str) -> None:
    class SoftmaxSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.softmax(input, 1)

    check(SoftmaxSimpleNet(), torch.randn(2, 3), target=target)


def test_softmax_dtype(target: str) -> None:
    class SoftmaxDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.softmax(input, 1, dtype=torch.float64)

    check(SoftmaxDtypeNet(), torch.randn(2, 3), target=target)
