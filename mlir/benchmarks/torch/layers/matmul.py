import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark

class MatmulNet(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.W)

def setup():
    weight = torch.randn(10000, 5000)
    model = MatmulNet(weight)
    x = torch.randn(8000, 10000)
    return model, x

if __name__ == "__main__":
    run_benchmark(setup, "matmul")