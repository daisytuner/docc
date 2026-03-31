import torch
import torch.nn as nn

import docc.torch


class Tail(nn.Module):
    """ResNet v2 tail: BN -> ReLU -> GlobalAvgPool -> Flatten -> FC.

    Takes the final feature map and produces class logits.
    """

    def __init__(self, in_channels=512, num_classes=1000):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def setup():
    """Return (eval-mode model, example_input) for the tail classifier."""
    model = Tail(in_channels=512, num_classes=1000)
    model.eval()
    x = torch.randn(1, 512, 7, 7)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_tail")
