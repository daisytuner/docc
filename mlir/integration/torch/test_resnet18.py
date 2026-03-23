import copy

import torch
import torchvision.models as models
import pytest

import docc.torch

def setup():
    """Return (eval-mode model, example_input) for ResNet-18 with random weights."""
    model = models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x

@pytest.mark.skip("Error: Not enough arguments provided")
def test_resnet18_compile():
    """docc backend output matches PyTorch eager output (default compile path)."""
    model, x = setup()
    model_ref = copy.deepcopy(model)

    program = docc.torch.compile_torch(model, x)
    with torch.no_grad():
        res = program(x)
        res_ref = model_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-5)

def test_resnet18_backend():
    """docc backend with target='none' output matches PyTorch eager output."""
    model, x = setup()
    model_ref = copy.deepcopy(model)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(x)
        res_ref = model_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-5)
