import torch
import torch.nn as nn
import pytest

import docc.torch

class SubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def setup():
    """Return (eval-mode model, example_input) for the full pipeline."""
    model = SubModel()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x

if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "bn_conv_bn_relu_maxpool")
