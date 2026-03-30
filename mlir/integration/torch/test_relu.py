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


# --- ReLU: 2-D ---


def test_relu_2d_compile():
    class ReLU2dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ReLU2dCompile().eval(), torch.randn(4, 64))


def test_relu_2d_backend():
    class ReLU2dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ReLU2dBackend().eval(), torch.randn(4, 64))


# --- ReLU: 4-D (conv-like) ---


def test_relu_4d_compile():
    class ReLU4dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ReLU4dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_relu_4d_backend():
    class ReLU4dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ReLU4dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- ReLU: large tensor ---


def test_relu_large_compile():
    class ReLULargeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ReLULargeCompile().eval(), torch.randn(16, 256))


def test_relu_large_backend():
    class ReLULargeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ReLULargeBackend().eval(), torch.randn(16, 256))


# --- ReLU6 ---


def test_relu6_2d_compile():
    class ReLU62dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ReLU62dCompile().eval(), torch.randn(4, 64))


def test_relu6_2d_backend():
    class ReLU62dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ReLU62dBackend().eval(), torch.randn(4, 64))


def test_relu6_4d_compile():
    class ReLU64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ReLU64dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_relu6_4d_backend():
    class ReLU64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ReLU64dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- LeakyReLU ---


def test_leaky_relu_default_compile():
    class LeakyReLUDefaultCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(LeakyReLUDefaultCompile().eval(), torch.randn(4, 64))


def test_leaky_relu_default_backend():
    class LeakyReLUDefaultBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(LeakyReLUDefaultBackend().eval(), torch.randn(4, 64))


def test_leaky_relu_slope_compile():
    class LeakyReLUSlopeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(LeakyReLUSlopeCompile().eval(), torch.randn(4, 64))


def test_leaky_relu_slope_backend():
    class LeakyReLUSlopeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(LeakyReLUSlopeBackend().eval(), torch.randn(4, 64))


# --- ELU ---


def test_elu_default_compile():
    class ELUDefaultCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ELUDefaultCompile().eval(), torch.randn(4, 64))


def test_elu_default_backend():
    class ELUDefaultBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ELUDefaultBackend().eval(), torch.randn(4, 64))


def test_elu_alpha_compile():
    class ELUAlphaCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU(alpha=0.5)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(ELUAlphaCompile().eval(), torch.randn(4, 64))


def test_elu_alpha_backend():
    class ELUAlphaBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU(alpha=0.5)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(ELUAlphaBackend().eval(), torch.randn(4, 64))


# --- GELU ---


def test_gelu_compile():
    class GELUCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(GELUCompile().eval(), torch.randn(4, 64))


def test_gelu_backend():
    class GELUBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(GELUBackend().eval(), torch.randn(4, 64))


def test_gelu_4d_compile():
    class GELU4dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(GELU4dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_gelu_4d_backend():
    class GELU4dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(GELU4dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- SiLU ---


def test_silu_compile():
    class SiLUCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_compile(SiLUCompile().eval(), torch.randn(4, 64))


def test_silu_backend():
    class SiLUBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    _check_backend(SiLUBackend().eval(), torch.randn(4, 64))
