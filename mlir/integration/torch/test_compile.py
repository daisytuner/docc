import pytest

import torch
import torch.nn as nn

import docc.torch


def test_inference():
    docc.torch.set_backend_options(target="none", category="server")

    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet()
    model.eval()
    model_ref = LinearNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(2, 4)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == (2, 2)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip(
    reason="Training requires support for multiple outputs in SDFG translation"
)
def test_training():
    docc.torch.set_backend_options(target="none", category="server")

    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet()
    model_ref = LinearNet()
    example_input = torch.randn(2, 4)
    target = torch.randn(2, 2)

    program = torch.compile(model, backend="docc")
    optimizer = torch.optim.SGD(program.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for _ in range(10):
        optimizer.zero_grad()
        res = program(example_input)
        loss = criterion(res, target)
        loss.backward()
        optimizer.step()
