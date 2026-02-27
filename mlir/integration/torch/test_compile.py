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
    model_ref = IdentityNet()
    example_input = torch.randn(2, 1)

    program = compile_torch(model, example_input)
    res = program(example_input)
    assert torch.allclose(res, model_ref(example_input), rtol=1e-5)


def test_tensor_const():
    class OutputNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            const = torch.Tensor([[10.0], [21.0], [3.0], [4.0]])
            return const

    model = OutputNet()
    model_ref = OutputNet()
    example_input = torch.randn(4, 1)

    program = compile_torch(model, example_input)
    res = program(example_input)
    assert torch.allclose(res, model_ref(example_input), rtol=1e-5)
