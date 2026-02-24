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