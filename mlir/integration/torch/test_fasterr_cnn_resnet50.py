import copy

import torch
import torchvision.models as models
import pytest

import docc.torch


def setup():
    """Return (eval-mode model, example_input) for Faster R-CNN ResNet50 with random weights."""
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x


def test_fasterrcnn_resnet50_backbone():
    """docc backend compiling only the backbone matches PyTorch eager output."""
    model, x = setup()
    model_ref = copy.deepcopy(model)

    docc.torch.set_backend_options(target="none", category="server")
    
    # Compile only the backbone with docc
    compiled_backbone = torch.compile(model.backbone, backend="docc")
    model.backbone = compiled_backbone
    
    with torch.no_grad():
        res = model([x.squeeze(0)])
        res_ref = model_ref([x.squeeze(0)])

    # Compare detection outputs
    assert len(res) == len(res_ref), "Number of outputs should match"
    assert torch.allclose(res[0]['boxes'], res_ref[0]['boxes'], rtol=1e-2, atol=1e-3)
    assert torch.allclose(res[0]['scores'], res_ref[0]['scores'], rtol=1e-2, atol=1e-3)
    assert torch.allclose(res[0]['labels'], res_ref[0]['labels'], rtol=1e-2, atol=1e-3)