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


def test_tensor_const():
    class OutputNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            const = torch.Tensor([[10.0], [21.0], [3.0], [4.0]])
            return const

    model = OutputNet()
    example_input = torch.randn(4, 1)
    ref_out = torch.Tensor([[10.0], [21.0], [3.0], [4.0]])

    program = compile_torch(model, example_input)
    res = program(example_input)
    assert torch.equal(res, ref_out)
