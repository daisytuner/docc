import copy
from xml.parsers.expat import model

from integration.torch.check import check_backend, check_compile
import torch
import torch.nn as nn

import docc.torch
from docc.torch import TorchProgram


class LinearNet(nn.Module):
    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


def test_remote_tuning_backend():
    """TorchProgram constructed with remote_tuning=True compiles and gives correct result."""
    model = LinearNet().eval()
    example_input = torch.randn(4, 8)

    check_backend(model, example_input, rtol=1e-4, atol=1e-5, target="sequential", category="server", remote_tuning=True)


def test_remote_tuning_compile_override():
    """remote_tuning=True passed directly to compile() compiles and gives correct result."""
    model = LinearNet().eval()

    example_input = torch.randn(4, 8)
    check_compile(model, example_input, rtol=1e-4, atol=1e-5, target="sequential", category="server", remote_tuning=True)
