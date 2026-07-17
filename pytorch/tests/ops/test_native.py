import torch
import torch.nn as nn

import docc.pytorch

# TODO: Currently not working


def test_add() -> None:
    class NativeAddNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: float, y: float) -> float:
            return x + y

    model = NativeAddNet()
    x = 0.3
    y = 0.4
    with torch.no_grad():
        program = torch.compile(model, backend="docc")
        res = program(x, y)
        ref = model(x, y)
    assert res == ref
