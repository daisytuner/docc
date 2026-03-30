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


# --- Softmax dim=-1 (3-D) ---


def test_softmax_last_dim_compile():
    class SoftmaxLastDimCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(SoftmaxLastDimCompile().eval(), torch.randn(4, 16, 64))


def test_softmax_last_dim_backend():
    class SoftmaxLastDimBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(SoftmaxLastDimBackend().eval(), torch.randn(4, 16, 64))


# --- Softmax dim=1 (2-D) ---


def test_softmax_dim1_compile():
    class SoftmaxDim1Compile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(SoftmaxDim1Compile().eval(), torch.randn(8, 32))


def test_softmax_dim1_backend():
    class SoftmaxDim1Backend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(SoftmaxDim1Backend().eval(), torch.randn(8, 32))


# --- Softmax dim=0 ---


def test_softmax_dim0_compile():
    class SoftmaxDim0Compile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=0)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(SoftmaxDim0Compile().eval(), torch.randn(16, 32))


def test_softmax_dim0_backend():
    class SoftmaxDim0Backend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=0)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(SoftmaxDim0Backend().eval(), torch.randn(16, 32))


# --- Softmax 4-D (channel dim) ---


def test_softmax_4d_compile():
    class Softmax4dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(Softmax4dCompile().eval(), torch.randn(2, 10, 8, 8))


def test_softmax_4d_backend():
    class Softmax4dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(Softmax4dBackend().eval(), torch.randn(2, 10, 8, 8))


# --- Large vocabulary-like softmax ---


def test_softmax_large_vocab_compile():
    class SoftmaxLargeVocabCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(SoftmaxLargeVocabCompile().eval(), torch.randn(4, 50000))


def test_softmax_large_vocab_backend():
    class SoftmaxLargeVocabBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(SoftmaxLargeVocabBackend().eval(), torch.randn(4, 50000))


# --- LogSoftmax ---


def test_logsoftmax_compile():
    class LogSoftmaxCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.LogSoftmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(LogSoftmaxCompile().eval(), torch.randn(4, 16, 64))


def test_logsoftmax_backend():
    class LogSoftmaxBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.LogSoftmax(dim=-1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(LogSoftmaxBackend().eval(), torch.randn(4, 16, 64))


def test_logsoftmax_dim1_compile():
    class LogSoftmaxDim1Compile(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.LogSoftmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_compile(LogSoftmaxDim1Compile().eval(), torch.randn(8, 32))


def test_logsoftmax_dim1_backend():
    class LogSoftmaxDim1Backend(nn.Module):
        def __init__(self):
            super().__init__()
            self.sm = nn.LogSoftmax(dim=1)

        def forward(self, x: torch.Tensor):
            return self.sm(x)

    _check_backend(LogSoftmaxDim1Backend().eval(), torch.randn(8, 32))
