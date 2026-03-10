import torch
import torch.nn as nn
import pytest

import docc.torch


# --- Basic 2D transpose ---


def test_2d_backend():
    class Transpose2dNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = Transpose2dNet()
    example_input = torch.randn(8, 10)

    model_ref = Transpose2dNet()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_2d_compile():
    class Transpose2dNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = Transpose2dNet()
    example_input = torch.randn(8, 10)

    model_ref = Transpose2dNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- .T property ---


def test_t_property_compile():
    class TPropertyNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.T

    model = TPropertyNet()
    example_input = torch.randn(8, 10)

    model_ref = TPropertyNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_t_property_backend():
    class TPropertyNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.T

    model = TPropertyNet()
    example_input = torch.randn(8, 10)

    model_ref = TPropertyNet()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Square matrix transpose ---


def test_square_compile():
    class SquareTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = SquareTransposeNet()
    example_input = torch.randn(10, 10)

    model_ref = SquareTransposeNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_square_backend():
    class SquareTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = SquareTransposeNet()
    example_input = torch.randn(10, 10)

    model_ref = SquareTransposeNet()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- 3D transpose (different dim pairs) ---


def test_3d_dim01_compile():
    class Transpose3dDim01Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = Transpose3dDim01Net()
    example_input = torch.randn(4, 8, 10)

    model_ref = Transpose3dDim01Net()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_3d_dim01_backend():
    class Transpose3dDim01Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = Transpose3dDim01Net()
    example_input = torch.randn(4, 8, 10)

    model_ref = Transpose3dDim01Net()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_3d_dim12_compile():
    class Transpose3dDim12Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 1, 2)

    model = Transpose3dDim12Net()
    example_input = torch.randn(5, 8, 10)

    model_ref = Transpose3dDim12Net()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_3d_dim12_backend():
    class Transpose3dDim12Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 1, 2)

    model = Transpose3dDim12Net()
    example_input = torch.randn(5, 8, 10)

    model_ref = Transpose3dDim12Net()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_3d_dim02_compile():
    class Transpose3dDim02Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 2)

    model = Transpose3dDim02Net()
    example_input = torch.randn(6, 8, 10)

    model_ref = Transpose3dDim02Net()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_3d_dim02_backend():
    class Transpose3dDim02Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 2)

    model = Transpose3dDim02Net()
    example_input = torch.randn(6, 8, 10)

    model_ref = Transpose3dDim02Net()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Double transpose (should be identity) ---


def test_double_transpose_compile():
    class DoubleTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(torch.transpose(x, 0, 1), 0, 1)

    model = DoubleTransposeNet()
    example_input = torch.randn(9, 10)

    model_ref = DoubleTransposeNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_double_transpose_backend():
    class DoubleTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(torch.transpose(x, 0, 1), 0, 1)

    model = DoubleTransposeNet()
    example_input = torch.randn(9, 10)

    model_ref = DoubleTransposeNet()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- .permute() ---


def test_permute_compile():
    class PermuteNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.permute(2, 0, 1)

    model = PermuteNet()
    example_input = torch.randn(7, 8, 10)

    model_ref = PermuteNet()

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_permute_backend():
    class PermuteNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.permute(2, 0, 1)

    model = PermuteNet()
    example_input = torch.randn(7, 8, 10)

    model_ref = PermuteNet()

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)
