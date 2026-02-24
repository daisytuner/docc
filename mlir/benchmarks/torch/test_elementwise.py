import torch
import torch.nn as nn

from docc.torch import compile_torch

def test_add():
    class AddNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(x, x)

    model = AddNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.add(example_input, example_input)
    assert torch.allclose(res, ref)
