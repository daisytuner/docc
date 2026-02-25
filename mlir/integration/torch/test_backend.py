import torch
import torch.nn as nn

import docc.torch

docc.torch.set_backend_options(target="none", category="server")


def test_torch_compile():
    class IdentityNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x

    model = IdentityNet()
    example_input = torch.randn(2, 1)

    program = torch.compile(model, backend="docc")
    res = program(example_input)
    assert torch.allclose(res, example_input)
