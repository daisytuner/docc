import torch
import torch.nn as nn

import docc.torch


class BasicBlock(nn.Module):
    """ResNet v2 basic block without downsampling (stage 1).

    BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) + residual
    """

    def __init__(self, channels=64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        identity = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x + identity


def setup():
    """Return (eval-mode model, example_input) for a basic residual block."""
    model = BasicBlock(channels=64)
    model.eval()
    x = torch.randn(1, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_basic_block")
