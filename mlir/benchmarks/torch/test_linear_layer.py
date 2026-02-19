import torch
import torch.nn as nn

from torch_mlir import fx


class LinearNet(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        h1 = self.linear1(x)
        h2 = self.linear2(h1)
        return h2


model = LinearNet(10, 16, 3)
example_input = torch.randn(8, 10)

torch_mlir = fx.export_and_import(
    model, example_input, output_type=fx.OutputType.LINALG_ON_TENSORS
)
print(str(torch_mlir))
