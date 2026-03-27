import torch
import torch.nn as nn
import pytest

import docc.torch


# --- Self matmul (x @ x) ---


def test_quadratic_self_backend():
    class SelfMatmulNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, x)
            return h1

    model = SelfMatmulNet()
    model_ref = SelfMatmulNet()

    example_input = torch.randn(10, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_quadratic_self_compile():
    class SelfMatmulNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, x)
            return h1

    model = SelfMatmulNet()
    example_input = torch.randn(10, 10)

    model_ref = SelfMatmulNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- torch.matmul with nn.Parameter weight ---


def test_parameter_weight_compile():
    class ParamWeightNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(10, 5)
    model = ParamWeightNet(weight)
    model_ref = ParamWeightNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_parameter_weight_backend():
    class ParamWeightNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(10, 5)
    model = ParamWeightNet(weight)
    model_ref = ParamWeightNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- @ operator ---


def test_at_operator_compile():
    class AtOperatorNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return x @ self.W

    weight = torch.randn(10, 5)
    model = AtOperatorNet(weight)
    model_ref = AtOperatorNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_at_operator_backend():
    class AtOperatorNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return x @ self.W

    weight = torch.randn(10, 5)
    model = AtOperatorNet(weight)
    model_ref = AtOperatorNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- torch.mm (strict 2D matmul) ---


def test_mm_compile():
    class MmNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.mm(x, self.W)

    weight = torch.randn(10, 5)
    model = MmNet(weight)
    model_ref = MmNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_mm_backend():
    class MmNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.mm(x, self.W)

    weight = torch.randn(10, 5)
    model = MmNet(weight)
    model_ref = MmNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- Non-square matrices ---


def test_non_square_compile():
    class NonSquareNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(32, 7)
    model = NonSquareNet(weight)
    model_ref = NonSquareNet(weight.clone())
    example_input = torch.randn(4, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_non_square_backend():
    class NonSquareNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(32, 7)
    model = NonSquareNet(weight)
    model_ref = NonSquareNet(weight.clone())
    example_input = torch.randn(4, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- Transposed weight (x @ W.T) ---


def test_transposed_weight_compile():
    class TransposedWeightNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W.T)

    weight = torch.randn(5, 10)
    model = TransposedWeightNet(weight)
    model_ref = TransposedWeightNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_transposed_weight_backend():
    class TransposedWeightNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W.T)

    weight = torch.randn(5, 10)
    model = TransposedWeightNet(weight)
    model_ref = TransposedWeightNet(weight.clone())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- Batched matmul (3D) ---


def test_batched_matmul_compile():
    class BatchedMatmulNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(4, 10, 5)
    model = BatchedMatmulNet(weight)
    model_ref = BatchedMatmulNet(weight.clone())
    example_input = torch.randn(4, 8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_batched_matmul_backend():
    class BatchedMatmulNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.matmul(x, self.W)

    weight = torch.randn(4, 10, 5)
    model = BatchedMatmulNet(weight)
    model_ref = BatchedMatmulNet(weight.clone())
    example_input = torch.randn(4, 8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- Chained matmul (x @ W1 @ W2) ---


def test_chained_matmul_compile():
    class ChainedMatmulNet(nn.Module):
        def __init__(self, w1: torch.Tensor, w2: torch.Tensor):
            super().__init__()
            self.W1 = nn.Parameter(w1)
            self.W2 = nn.Parameter(w2)

        def forward(self, x: torch.Tensor):
            return torch.matmul(torch.matmul(x, self.W1), self.W2)

    w1 = torch.randn(10, 16)
    w2 = torch.randn(16, 3)
    model = ChainedMatmulNet(w1, w2)
    model_ref = ChainedMatmulNet(w1.clone(), w2.clone())
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_chained_matmul_backend():
    class ChainedMatmulNet(nn.Module):
        def __init__(self, w1: torch.Tensor, w2: torch.Tensor):
            super().__init__()
            self.W1 = nn.Parameter(w1)
            self.W2 = nn.Parameter(w2)

        def forward(self, x: torch.Tensor):
            return torch.matmul(torch.matmul(x, self.W1), self.W2)

    w1 = torch.randn(10, 16)
    w2 = torch.randn(16, 3)
    model = ChainedMatmulNet(w1, w2)
    model_ref = ChainedMatmulNet(w1.clone(), w2.clone())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


# --- Expand / Collapse (reshape around matmul) ---


def test_expand_collapse_compile():
    """Flatten a 3D input to 2D, matmul, then reshape back to 3D."""

    class ExpandCollapseNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            B, S, D = x.shape
            flat = x.reshape(B * S, D)  # collapse: 3D -> 2D
            out = torch.matmul(flat, self.W)
            return out.reshape(B, S, -1)  # expand: 2D -> 3D

    weight = torch.randn(16, 8)
    model = ExpandCollapseNet(weight)
    model_ref = ExpandCollapseNet(weight.clone())
    example_input = torch.randn(4, 6, 16)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)


def test_expand_collapse_backend():
    """Flatten a 3D input to 2D, matmul, then reshape back to 3D."""

    class ExpandCollapseNet(nn.Module):
        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.W = nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            B, S, D = x.shape
            flat = x.reshape(B * S, D)
            out = torch.matmul(flat, self.W)
            return out.reshape(B, S, -1)

    weight = torch.randn(16, 8)
    model = ExpandCollapseNet(weight)
    model_ref = ExpandCollapseNet(weight.clone())
    example_input = torch.randn(4, 6, 16)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4)
