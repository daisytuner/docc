import torch
import torch.nn as nn

from tests import check


def test_relu(target: str) -> None:
    class ReLUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu: nn.ReLU = nn.ReLU()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.relu(input)

    check(ReLUNet(), torch.randn(2), target=target)
