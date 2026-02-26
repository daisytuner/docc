import pytest

import torch
import torch.nn as nn

from docc.torch import compile_torch

def test_add():
    class AddNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(x, x)

    model = AddNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.add(example_input, example_input)
    assert torch.allclose(res, ref)

def test_add2():
    class Add2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(torch.add(x, x), x)

    model = Add2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.add(torch.add(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

def test_sub():
    class SubNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sub(x, x)

    model = SubNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.sub(example_input, example_input)
    assert torch.allclose(res, ref)

def test_sub2():
    class Sub2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sub(torch.sub(x, x), x)

    model = Sub2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.sub(torch.sub(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

def test_mul():
    class MulNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mul(x, x)

    model = MulNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.mul(example_input, example_input)
    assert torch.allclose(res, ref)

def test_mul2():
    class Mul2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mul(torch.mul(x, x), x)

    model = Mul2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.mul(torch.mul(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

def test_div():
    class DivNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.div(x, x)

    model = DivNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.div(example_input, example_input)
    assert torch.allclose(res, ref)

def test_div2():
    class Div2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.div(torch.div(x, x), x)

    model = Div2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.div(torch.div(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_abs():
    class AbsNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.abs(x)
    
    model = AbsNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.abs(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_abs2():
    class Abs2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.abs(torch.abs(x))
    
    model = Abs2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.abs(torch.abs(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_ceil():
    class CeilNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ceil(x)
    
    model = CeilNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.ceil(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_ceil2():
    class Ceil2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ceil(torch.ceil(x))
    
    model = Ceil2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.ceil(torch.ceil(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_erf():
    class ErfNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erf(x)
    
    model = ErfNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.erf(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_erf2():
    class Erf2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erf(torch.erf(x))
    
    model = Erf2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.erf(torch.erf(example_input))
    assert torch.allclose(res, ref)

def test_exp():
    class ExpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp(x)
    
    model = ExpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.exp(example_input)
    assert torch.allclose(res, ref)

def test_exp2():
    class Exp2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp(torch.exp(x))
    
    model = Exp2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.exp(torch.exp(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_floor():
    class FloorNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor(x)
    
    model = FloorNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.floor(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_floor2():
    class Floor2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor(torch.floor(x))
    
    model = Floor2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.floor(torch.floor(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_log():
    class LogNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log(x)
    
    model = LogNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.log(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_log2():
    class Log2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log(torch.log(x))
    
    model = Log2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.log(torch.log(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_max():
    class MaxNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.max(x, x)
    
    model = MaxNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.max(example_input, example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_max2():
    class Max2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.max(torch.max(x, x), x)
    
    model = Max2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.max(torch.max(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_min():
    class MinNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.min(x, x)
    
    model = MinNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.min(example_input, example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_min2():
    class Min2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.min(torch.min(x, x), x)
    
    model = Min2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.min(torch.min(example_input, example_input), example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_pow():
    class PowNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.pow(x, 3)
    
    model = PowNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.pow(example_input, 3)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_pow2():
    class Pow2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.pow(torch.pow(x, 2), 2)
    
    model = Pow2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.powf(torch.powf(example_input, 2), 2)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_round():
    class RoundNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.round(x)
    
    model = RoundNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.round(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_round2():
    class Round2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.round(torch.round(x))
    
    model = Round2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.round(torch.round(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_sqrt():
    class SqrtNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sqrt(x)
    
    model = SqrtNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.sqrt(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_sqrt2():
    class Sqrt2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sqrt(torch.sqrt(x))
    
    model = Sqrt2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.sqrt(torch.sqrt(example_input))
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_tanh():
    class TanhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tanh(x)
    
    model = TanhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.tanh(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip()
def test_tanh2():
    class Tanh2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tanh(torch.tanh(x))
    
    model = Tanh2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = torch.tanh(torch.tanh(example_input))
    assert torch.allclose(res, ref)
