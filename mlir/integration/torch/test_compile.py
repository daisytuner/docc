import torch
import torch.nn as nn

from docc.torch import compile_torch


def test_identitfy():
    class IdentityNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x

    model = IdentityNet()
    example_input = torch.randn(2, 1)

    program = compile_torch(model, example_input)
    res = program(example_input)
    assert torch.allclose(res, example_input)
