import torch
import torch.nn as nn

import docc.torch


class BNReluConv(nn.Module):
    """Single pre-activation conv unit: BN -> ReLU -> Conv(3x3).

    The smallest non-trivial building block of ResNet v2.
    """

    def __init__(self, in_channels=64, out_channels=64, stride=1, padding=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


def setup():
    """Return (eval-mode model, example_input) for BN -> ReLU -> Conv."""
    model = BNReluConv(in_channels=64, out_channels=64)
    model.eval()
    x = torch.randn(1, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_bn_relu_conv")
