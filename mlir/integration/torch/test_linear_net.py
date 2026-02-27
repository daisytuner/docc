import torch
import torch.nn as nn

import docc.torch


def test_pytorch():
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
    model_ref = LinearNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model)
    res = program(example_input)

    res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_backend():
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
    model_ref = LinearNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    res = program(example_input)

    res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_compile():
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
    model_ref = LinearNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    res = program(example_input)

    res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, rtol=1e-4)
