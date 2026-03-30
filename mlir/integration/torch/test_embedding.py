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


# --- Small vocabulary ---


def test_embedding_small_compile():
    class EmbSmallCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 64)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(EmbSmallCompile().eval(), torch.randint(0, 256, (4, 16)))


def test_embedding_small_backend():
    class EmbSmallBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 64)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(EmbSmallBackend().eval(), torch.randint(0, 256, (4, 16)))


# --- Large vocabulary ---


def test_embedding_large_vocab_compile():
    class EmbLargeVocabCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(30000, 128)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(EmbLargeVocabCompile().eval(), torch.randint(0, 30000, (2, 32)))


def test_embedding_large_vocab_backend():
    class EmbLargeVocabBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(30000, 128)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(EmbLargeVocabBackend().eval(), torch.randint(0, 30000, (2, 32)))


# --- 1-D input ---


def test_embedding_1d_compile():
    class Emb1dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 32)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(Emb1dCompile().eval(), torch.randint(0, 100, (8,)))


def test_embedding_1d_backend():
    class Emb1dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 32)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(Emb1dBackend().eval(), torch.randint(0, 100, (8,)))


# --- 3-D input ---


def test_embedding_3d_compile():
    class Emb3dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(512, 64)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(Emb3dCompile().eval(), torch.randint(0, 512, (2, 4, 8)))


def test_embedding_3d_backend():
    class Emb3dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(512, 64)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(Emb3dBackend().eval(), torch.randint(0, 512, (2, 4, 8)))


# --- With padding_idx ---


def test_embedding_padding_idx_compile():
    class EmbPadIdxCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 64, padding_idx=0)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(EmbPadIdxCompile().eval(), torch.randint(0, 256, (4, 16)))


def test_embedding_padding_idx_backend():
    class EmbPadIdxBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 64, padding_idx=0)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(EmbPadIdxBackend().eval(), torch.randint(0, 256, (4, 16)))


# --- Small embedding dim ---


def test_embedding_small_dim_compile():
    class EmbSmallDimCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 8)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_compile(EmbSmallDimCompile().eval(), torch.randint(0, 50, (4, 10)))


def test_embedding_small_dim_backend():
    class EmbSmallDimBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 8)

        def forward(self, x: torch.Tensor):
            return self.emb(x)

    _check_backend(EmbSmallDimBackend().eval(), torch.randint(0, 50, (4, 10)))
