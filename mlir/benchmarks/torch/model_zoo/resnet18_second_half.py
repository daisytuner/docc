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


class ResNet18SecondHalf(nn.Module):
    """ResNet18-v2 second half: Stage 3 + Stage 4 + Tail.

    Stage 3: DownsampleBlock(128->256) + BasicBlock(256)
    Stage 4: DownsampleBlock(256->512) + BasicBlock(512)
    Tail:    BN -> ReLU -> GlobalAvgPool -> Flatten -> FC(1000)

    Input:  (N, 128, 28, 28)
    Output: (N, 1000)
    """

    def __init__(self):
        super().__init__()
        # Stage 3
        self.s3_down = DownsampleBlock(128, 256)
        self.s3_block = BasicBlock(256)
        # Stage 4
        self.s4_down = DownsampleBlock(256, 512)
        self.s4_block = BasicBlock(512)
        # Tail
        self.bn_tail = nn.BatchNorm2d(512)
        self.relu_tail = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        # Stage 3
        x = self.s3_block(self.s3_down(x))
        # Stage 4
        x = self.s4_block(self.s4_down(x))
        # Tail
        x = self.relu_tail(self.bn_tail(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def setup():
    """Return (eval-mode model, example_input) for the second half of ResNet18."""
    model = ResNet18SecondHalf()
    model.eval()
    x = torch.randn(1, 128, 28, 28)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_second_half")
