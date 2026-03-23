import copy

import torch
import torch.nn as nn
import pytest

import docc.torch

LAYER_NAMES = ["bn1", "conv", "bn2", "relu", "maxpool"]

class SubModel(nn.Module):
    """A model built from a contiguous slice [start, end) of the full pipeline."""

    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

        if start <= 0 < end:
            self.bn1 = nn.BatchNorm2d(3)
        if start <= 1 < end:
            self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if start <= 2 < end:
            self.bn2 = nn.BatchNorm2d(64)
        if start <= 3 < end:
            self.relu = nn.ReLU()
        if start <= 4 < end:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.start <= 0 < self.end:
            x = self.bn1(x)
        if self.start <= 1 < self.end:
            x = self.conv(x)
        if self.start <= 2 < self.end:
            x = self.bn2(x)
        if self.start <= 3 < self.end:
            x = self.relu(x)
        if self.start <= 4 < self.end:
            x = self.maxpool(x)
        return x

def _input_shape(start: int):
    """Return the correct input shape for a sub-model starting at `start`."""
    if start <= 1:
        # BN1 or Conv expect 3-channel 224×224 input
        return (1, 3, 224, 224)
    else:
        # BN2, ReLU, MaxPool expect 64-channel 112×112 (post-conv shape)
        return (1, 64, 112, 112)

def _make_test_id(start: int, end: int):
    return "_".join(LAYER_NAMES[start:end])

# All contiguous subsequences (start, end) with start < end
_ALL_SUBSEQUENCES = [(s, e) for s in range(5) for e in range(s + 1, 6)]

@pytest.mark.skip("Error: Not enough arguments provided")
@pytest.mark.parametrize(
    "start,end",
    _ALL_SUBSEQUENCES,
    ids=[_make_test_id(s, e) for s, e in _ALL_SUBSEQUENCES],
)
def test_submodel_compile(start, end):
    """docc backend output matches PyTorch eager output (default compile)."""
    torch._dynamo.reset()

    model = SubModel(start, end)
    model.eval()
    x = torch.randn(*_input_shape(start))

    model_ref = copy.deepcopy(model)

    program = docc.torch.compile_torch(model, x)
    with torch.no_grad():
        res = program(x)
        res_ref = model_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-5), (
        f"Max abs diff: {(res - res_ref).abs().max().item():.6e}, "
        f"Max rel diff: {((res - res_ref).abs() / (res_ref.abs() + 1e-8)).max().item():.6e}"
    )

@pytest.mark.parametrize(
    "start,end",
    _ALL_SUBSEQUENCES,
    ids=[_make_test_id(s, e) for s, e in _ALL_SUBSEQUENCES],
)
def test_submodel_backend(start, end):
    """docc backend with target='none' matches PyTorch eager output."""
    torch._dynamo.reset()

    model = SubModel(start, end)
    model.eval()
    x = torch.randn(*_input_shape(start))

    model_ref = copy.deepcopy(model)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(x)
        res_ref = model_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-5), (
        f"Max abs diff: {(res - res_ref).abs().max().item():.6e}, "
        f"Max rel diff: {((res - res_ref).abs() / (res_ref.abs() + 1e-8)).max().item():.6e}"
    )
