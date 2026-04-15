import copy

import torch
import torchvision.models as models
import pytest

import docc.torch

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

def setup_yolo():
    """Return (eval-mode model, example_input) for YOLOv8n with random weights."""
    # YOLOv8n (nano) is the smallest/fastest variant
    yolo = YOLO('yolov8n.pt')
    model = yolo.model
    model.eval()
    x = torch.randn(1, 3, 640, 640)  # YOLO typically uses 640x640
    return model, x


def compare_outputs(res, res_ref, rtol=1e-3, atol=1e-4):
    """Recursively compare outputs that may be nested structures of tensors."""
    if isinstance(res, torch.Tensor):
        assert torch.allclose(res, res_ref, rtol=rtol, atol=atol)
    elif isinstance(res, dict):
        assert res.keys() == res_ref.keys(), "Dictionary keys don't match"
        for key in res.keys():
            compare_outputs(res[key], res_ref[key], rtol, atol)
    elif isinstance(res, (list, tuple)):
        assert len(res) == len(res_ref), f"Length mismatch: {len(res)} vs {len(res_ref)}"
        for r, r_ref in zip(res, res_ref):
            compare_outputs(r, r_ref, rtol, atol)
    else:
        # For other types, use standard equality
        assert res == res_ref

@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_yolov8_backbone_compile():
    """docc backend compiling only the backbone matches PyTorch eager output for YOLOv8."""
    model, x = setup_yolo()
    backbone_ref = copy.deepcopy(model.model[0])

    # Compile only the backbone with docc
    compiled_backbone = docc.torch.compile_torch(model.model[0], x)

    with torch.no_grad():
        res = compiled_backbone(x)
        res_ref = backbone_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_yolov8_backbone():
    """docc backend compiling only the backbone matches PyTorch eager output for YOLOv8."""
    model, x = setup_yolo()
    model_ref = copy.deepcopy(model)

    docc.torch.set_backend_options(target="none", category="server")
    
    # Compile only the backbone with docc
    compiled_backbone = torch.compile(model.model[0], backend="docc")
    model.model[0] = compiled_backbone
    
    with torch.no_grad():
        res = model(x)
        res_ref = model_ref(x)

    # Use recursive comparison to handle nested structures
    compare_outputs(res, res_ref, rtol=1e-3, atol=1e-4)
