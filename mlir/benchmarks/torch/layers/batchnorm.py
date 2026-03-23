import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark


class BatchNorm2dNet(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor):
        return self.bn(x)


# batch=64, channels=256, height=56, width=56
def setup():
    model = BatchNorm2dNet(256)
    model.eval()
    x = torch.randn(64, 256, 56, 56)
    return model, x


if __name__ == "__main__":
    run_benchmark(setup, "batchnorm2d")
