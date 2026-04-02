import torch
import torch.nn as nn

import docc.torch


class Stage1(nn.Module):
    """ResNet v2 stage 1: two basic blocks (no downsampling, 64 channels).

    Each block: BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) + skip
    """

    def __init__(self):
        super().__init__()
        # Block 1
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Block 2
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Block 1
        identity = x
        out = self.bn1_1(x)
        out = self.relu1_1(out)
        out = self.conv1_1(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_2(out)
        x = out + identity
        # Block 2
        identity = x
        out = self.bn2_1(x)
        out = self.relu2_1(out)
        out = self.conv2_1(out)
        out = self.bn2_2(out)
        out = self.relu2_2(out)
        out = self.conv2_2(out)
        return out + identity


def setup():
    """Return (eval-mode model, example_input) for stage 1."""
    model = Stage1()
    model.eval()
    x = torch.randn(1, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_stage1")
