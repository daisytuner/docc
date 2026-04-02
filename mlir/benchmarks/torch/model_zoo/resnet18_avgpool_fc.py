import torch
import torch.nn as nn

import docc.torch


class GlobalAvgPoolFC(nn.Module):
    """Global average pooling followed by a fully-connected layer.

    Tests the classifier head in isolation: AvgPool -> Flatten -> Linear.
    """

    def __init__(self, in_channels=512, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def setup():
    """Return (eval-mode model, example_input) for avgpool + FC."""
    model = GlobalAvgPoolFC(in_channels=512, num_classes=1000)
    model.eval()
    x = torch.randn(1, 512, 7, 7)
    return model, x


if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18_avgpool_fc")
