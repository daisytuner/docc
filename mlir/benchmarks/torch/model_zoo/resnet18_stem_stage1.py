import torch
import torch.nn as nn

import docc.torch


class StemPlusStage1(nn.Module):
    """ResNet v2 stem + stage 1: the first half of the network.

    Stem:   BN -> Conv(7x7,s2,p3) -> BN -> ReLU -> MaxPool(3x3,s2,p1)
    Stage1: 2x basic blocks [BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) + skip]
    """

    def __init__(self):
        super().__init__()
        # Stem
        self.bn0 = nn.BatchNorm2d(3)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stage 1 - Block 1
        self.s1_bn1_1 = nn.BatchNorm2d(64)
        self.s1_relu1_1 = nn.ReLU()
        self.s1_conv1_1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.s1_bn1_2 = nn.BatchNorm2d(64)
        self.s1_relu1_2 = nn.ReLU()
        self.s1_conv1_2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Stage 1 - Block 2
        self.s1_bn2_1 = nn.BatchNorm2d(64)
        self.s1_relu2_1 = nn.ReLU()
        self.s1_conv2_1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.s1_bn2_2 = nn.BatchNorm2d(64)
        self.s1_relu2_2 = nn.ReLU()
        self.s1_conv2_2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        # Stem
        x = self.bn0(x)
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        # Stage 1 - Block 1
        identity = x
        out = self.s1_bn1_1(x)
        out = self.s1_relu1_1(out)
        out = self.s1_conv1_1(out)
        out = self.s1_bn1_2(out)
        out = self.s1_relu1_2(out)
        out = self.s1_conv1_2(out)
        x = out + identity
        # Stage 1 - Block 2
        identity = x
        out = self.s1_bn2_1(x)
        out = self.s1_relu2_1(out)
        out = self.s1_conv2_1(out)
        out = self.s1_bn2_2(out)
        out = self.s1_relu2_2(out)
        out = self.s1_conv2_2(out)
        return out + identity


def setup():
    """Return (eval-mode model, example_input) for stem + stage 1."""
    model = StemPlusStage1()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_stem_stage1")
