import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark

class LinearNet(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

def setup():
    model = LinearNet(10000, 5000)
    x = torch.randn(8000, 10000)
    return model, x

if __name__ == "__main__":
    run_benchmark(setup, "linear")