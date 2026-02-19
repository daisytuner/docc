import torch
import torch.nn as nn

from torch_mlir import fx


class MatmulNet(nn.Module):
    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor):
        super().__init__()
        self.W1 = nn.Parameter(weight1)
        self.W2 = nn.Parameter(weight2)

    def forward(self, x: torch.Tensor):
        h1 = torch.matmul(x, self.W1)
        h2 = torch.matmul(h1, self.W2)
        return h2


weight1 = torch.randn(10, 16)
weight2 = torch.randn(16, 3)
model = MatmulNet(weight1, weight2)
example_input = torch.randn(8, 10)

torch_mlir = fx.export_and_import(
    model, example_input, output_type=fx.OutputType.LINALG_ON_TENSORS
)
print(str(torch_mlir))
