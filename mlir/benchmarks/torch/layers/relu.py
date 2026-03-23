import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark


class ReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(x)


# batch=64, channels=256, height=56, width=56
def setup():
    model = ReLUNet()
    x = torch.randn(64, 256, 56, 56)
    return model, x


if __name__ == "__main__":
    run_benchmark(setup, "relu")
