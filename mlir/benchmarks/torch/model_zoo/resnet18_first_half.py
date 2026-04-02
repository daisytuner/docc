import torch
import torch.nn as nn

import docc.torch


class BasicBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + identity


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        main = self.conv2(self.relu2(self.bn2(self.conv1(x))))
        return main + self.shortcut(x)


class ResNet18FirstHalf(nn.Module):
    """ResNet18-v2 first half: Stem + Stage 1 + Stage 2.

    Stem:    BN -> Conv(7x7,s2,p3) -> BN -> ReLU -> MaxPool(3x3,s2,p1)
    Stage 1: 2x BasicBlock(64)
    Stage 2: DownsampleBlock(64->128) + BasicBlock(128)

    Input:  (N, 3, 224, 224)
    Output: (N, 128, 28, 28)
    """

    def __init__(self):
        super().__init__()
        # Stem
        self.bn0 = nn.BatchNorm2d(3)
        self.conv0 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2, 1)
        # Stage 1
        self.s1_block1 = BasicBlock(64)
        self.s1_block2 = BasicBlock(64)
        # Stage 2
        self.s2_down = DownsampleBlock(64, 128)
        self.s2_block = BasicBlock(128)

    def forward(self, x):
        # Stem
        x = self.pool(self.relu0(self.bn1(self.conv0(self.bn0(x)))))
        # Stage 1
        x = self.s1_block2(self.s1_block1(x))
        # Stage 2
        x = self.s2_block(self.s2_down(x))
        return x


def setup():
    """Return (eval-mode model, example_input) for the first half of ResNet18."""
    model = ResNet18FirstHalf()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_first_half")
