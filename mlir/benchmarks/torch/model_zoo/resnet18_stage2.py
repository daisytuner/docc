import torch
import torch.nn as nn

import docc.torch


class Stage2(nn.Module):
    """ResNet v2 stage 2: downsample block + basic block (64 -> 128 channels).

    Block 1 (downsample): BN -> ReLU -> Conv(3x3,s2) -> BN -> ReLU -> Conv(3x3) + 1x1 shortcut
    Block 2 (basic):      BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) + skip
    """

    def __init__(self):
        super().__init__()
        # Downsample block
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Conv2d(
            64, 128, kernel_size=1, stride=2, padding=0, bias=False
        )
        # Basic block
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        # Downsample block
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        main = self.conv1_1(x)
        main = self.bn1_2(main)
        main = self.relu1_2(main)
        main = self.conv1_2(main)
        skip = self.shortcut(x)
        x = main + skip
        # Basic block
        identity = x
        out = self.bn2_1(x)
        out = self.relu2_1(out)
        out = self.conv2_1(out)
        out = self.bn2_2(out)
        out = self.relu2_2(out)
        out = self.conv2_2(out)
        return out + identity


def setup():
    """Return (eval-mode model, example_input) for stage 2."""
    model = Stage2()
    model.eval()
    x = torch.randn(1, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_stage2")
