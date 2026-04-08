import torch
import torchvision.models as models
import pytest

import docc.torch

def setup():
    """Return (eval-mode model, example_input) for ResNet-18 with random weights."""
    model = models.resnet18(weights=None)
    model.eval()
    x = torch.randn(32, 3, 224, 224)
    return model, x

if __name__ == "__main__":
    from benchmarks.harness import run_benchmark

    run_benchmark(setup, "resnet18")
