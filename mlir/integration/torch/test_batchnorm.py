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


# --- BatchNorm1d ---


def test_batchnorm1d_compile():
    class BN1dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN1dCompile().eval(), torch.randn(8, 32))


def test_batchnorm1d_backend():
    class BN1dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN1dBackend().eval(), torch.randn(8, 32))


def test_batchnorm1d_large_compile():
    class BN1dLargeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(256)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN1dLargeCompile().eval(), torch.randn(32, 256))


def test_batchnorm1d_large_backend():
    class BN1dLargeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(256)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN1dLargeBackend().eval(), torch.randn(32, 256))


def test_batchnorm1d_no_affine_compile():
    class BN1dNoAffineCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32, affine=False)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN1dNoAffineCompile().eval(), torch.randn(8, 32))


def test_batchnorm1d_no_affine_backend():
    class BN1dNoAffineBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32, affine=False)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN1dNoAffineBackend().eval(), torch.randn(8, 32))


def test_batchnorm1d_3d_input_compile():
    """BatchNorm1d also accepts (N, C, L) inputs."""

    class BN1d3dInputCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(16)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN1d3dInputCompile().eval(), torch.randn(4, 16, 10))


def test_batchnorm1d_3d_input_backend():
    class BN1d3dInputBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(16)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN1d3dInputBackend().eval(), torch.randn(4, 16, 10))


# --- BatchNorm2d ---


def test_batchnorm2d_compile():
    class BN2dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2dCompile().eval(), torch.randn(4, 64, 16, 16))


def test_batchnorm2d_backend():
    class BN2dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2dBackend().eval(), torch.randn(4, 64, 16, 16))


def test_batchnorm2d_small_spatial_compile():
    class BN2dSmallSpatialCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(128)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2dSmallSpatialCompile().eval(), torch.randn(8, 128, 4, 4))


def test_batchnorm2d_small_spatial_backend():
    class BN2dSmallSpatialBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(128)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2dSmallSpatialBackend().eval(), torch.randn(8, 128, 4, 4))


def test_batchnorm2d_no_affine_compile():
    class BN2dNoAffineCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64, affine=False)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2dNoAffineCompile().eval(), torch.randn(4, 64, 8, 8))


def test_batchnorm2d_no_affine_backend():
    class BN2dNoAffineBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64, affine=False)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2dNoAffineBackend().eval(), torch.randn(4, 64, 8, 8))


# --- BatchNorm2d (ResNet stem shapes) ---


def test_batchnorm2d_3ch_compile():
    """BatchNorm2d(3) on (1, 3, 224, 224) — matches ResNet stem BN1."""

    class BN2d3chCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2d3chCompile().eval(), torch.randn(1, 3, 224, 224))


def test_batchnorm2d_3ch_backend():
    class BN2d3chBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2d3chBackend().eval(), torch.randn(1, 3, 224, 224))


def test_batchnorm2d_64ch_large_spatial_compile():
    """BatchNorm2d(64) on (1, 64, 112, 112) — matches ResNet stem BN2."""

    class BN2d64chLargeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2d64chLargeCompile().eval(), torch.randn(1, 64, 112, 112))


def test_batchnorm2d_64ch_large_spatial_backend():
    class BN2d64chLargeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(64)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2d64chLargeBackend().eval(), torch.randn(1, 64, 112, 112))


def test_batchnorm2d_batch1_compile():
    """BatchNorm2d with batch_size=1 (edge case for running stats)."""

    class BN2dBatch1Compile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN2dBatch1Compile().eval(), torch.randn(1, 16, 32, 32))


def test_batchnorm2d_batch1_backend():
    class BN2dBatch1Backend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN2dBatch1Backend().eval(), torch.randn(1, 16, 32, 32))


# --- BatchNorm3d ---


def test_batchnorm3d_compile():
    class BN3dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm3d(32)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_compile(BN3dCompile().eval(), torch.randn(2, 32, 4, 4, 4))


def test_batchnorm3d_backend():
    class BN3dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm3d(32)

        def forward(self, x: torch.Tensor):
            return self.bn(x)

    _check_backend(BN3dBackend().eval(), torch.randn(2, 32, 4, 4, 4))
