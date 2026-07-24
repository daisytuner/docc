import torch
import torch.nn as nn

from tests import check

# --- Linear ---


def test_linear_simple(target: str) -> None:
    class LinearSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    check(LinearSimpleNet(), torch.randn(128, 20), target=target)


def test_linear_bias(target: str) -> None:
    class LinearBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    check(LinearBiasNet(), torch.randn(128, 20), target=target)
