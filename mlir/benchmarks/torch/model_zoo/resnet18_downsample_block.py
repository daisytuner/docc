import torch
import torch.nn as nn

import docc.torch


class DownsampleBlock(nn.Module):
    """ResNet v2 basic block with downsampling (stages 2-4 first block).

    Main path:  BN -> ReLU -> Conv(3x3,s2) -> BN -> ReLU -> Conv(3x3,s1)
    Shortcut:   1x1 Conv(s2) from the activated input
    Output:     main + shortcut
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False
        )

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        main = self.conv1(x)
        main = self.bn2(main)
        main = self.relu2(main)
        main = self.conv2(main)
        skip = self.shortcut(x)
        return main + skip


def setup():
    """Return (eval-mode model, example_input) for a downsample block (64->128)."""
    model = DownsampleBlock(in_channels=64, out_channels=128)
    model.eval()
    x = torch.randn(1, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_downsample_block")
