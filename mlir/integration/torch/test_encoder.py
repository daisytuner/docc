import copy

import pytest

import torch
import torch.nn as nn

import docc.torch


# --- helpers ---


def _check(model, example_input, rtol=1e-4, atol=1e-5):
    model_ref = copy.deepcopy(model)
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=rtol, atol=atol)


def _check_compile(model, example_input, rtol=1e-4, atol=1e-5):
    model_ref = copy.deepcopy(model)
    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=rtol, atol=atol)


def _check_backend(model, example_input, rtol=1e-4, atol=1e-5):
    docc.torch.set_backend_options(target="none", category="server")
    _check(model, example_input, rtol=rtol, atol=atol)


# --- Single encoder layer: 8 heads, d_model=64, ff=256 ---
@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_compile():
    class EncLayer8h64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=True,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_compile(EncLayer8h64dCompile().eval(), torch.randn(2, 16, 64))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_backend():
    class EncLayer8h64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=True,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_backend(EncLayer8h64dBackend().eval(), torch.randn(2, 16, 64))


# --- Single encoder layer: 4 heads, d_model=128, ff=512 ---


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_4h128d_compile():
    class EncLayer4h128dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_compile(EncLayer4h128dCompile().eval(), torch.randn(2, 32, 128))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_4h128d_backend():
    class EncLayer4h128dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_backend(EncLayer4h128dBackend().eval(), torch.randn(2, 32, 128))


# --- Single encoder layer: norm_first=False (post-norm) ---


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_postnorm_compile():
    class EncLayerPostnormCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=False,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_compile(EncLayerPostnormCompile().eval(), torch.randn(2, 16, 64))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_encoder_layer_postnorm_backend():
    class EncLayerPostnormBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=False,
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    _check_backend(EncLayerPostnormBackend().eval(), torch.randn(2, 16, 64))


# --- Stacked encoder: 2 layers ---


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_2layer_compile():
    class TFEnc2LayerCompile(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_compile(TFEnc2LayerCompile().eval(), torch.randn(2, 16, 64))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_2layer_backend():
    class TFEnc2LayerBackend(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_backend(TFEnc2LayerBackend().eval(), torch.randn(2, 16, 64))


# --- Stacked encoder: 4 layers, larger dims ---


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_4layer_compile():
    class TFEnc4LayerCompile(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=4)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_compile(TFEnc4LayerCompile().eval(), torch.randn(1, 16, 128))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_4layer_backend():
    class TFEnc4LayerBackend(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=4)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_backend(TFEnc4LayerBackend().eval(), torch.randn(1, 16, 128))


# --- Stacked encoder with post-norm ---


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_postnorm_compile():
    class TFEncPostnormCompile(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_compile(TFEncPostnormCompile().eval(), torch.randn(2, 16, 64))


@pytest.mark.skip(reason="Unsupported by torch-mlir")
def test_transformer_encoder_postnorm_backend():
    class TFEncPostnormBackend(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x: torch.Tensor):
            return self.encoder(x)

    _check_backend(TFEncPostnormBackend().eval(), torch.randn(2, 16, 64))
