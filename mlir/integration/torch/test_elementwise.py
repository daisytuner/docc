import pytest

import torch
import torch.nn as nn

from docc.torch import compile_torch

@pytest.mark.skip("Requires math dialect")
def test_abs():
    class AbsNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.abs(x)

    model = AbsNet()
    model_ref = AbsNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_absolute():
    class AbsoluteNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.absolute(x)

    model = AbsoluteNet()
    model_ref = AbsoluteNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_acos():
    class AcosNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.acos(x)

    model = AcosNet()
    model_ref = AcosNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arccos():
    class ArccosNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arccos(x)

    model = ArccosNet()
    model_ref = ArccosNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_acosh():
    class AcoshNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.acosh(x)

    model = AcoshNet()
    model_ref = AcoshNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arccosh():
    class ArccoshNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arccosh(x)

    model = ArccoshNet()
    model_ref = ArccoshNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_add():
    class AddNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(x, x)

    model = AddNet()
    model_ref = AddNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_addcdiv():
    class AddcdivNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.addcdiv(x, x, x, value=2)

    model = AddcdivNet()
    model_ref = AddcdivNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_addcmul():
    class AddcmulNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.addcmul(x, x, x, value=2)

    model = AddcmulNet()
    model_ref = AddcmulNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_angle():
    class AngleNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.angle(x)

    model = AngleNet()
    model_ref = AngleNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_asin():
    class AsinNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.asin(x)

    model = AsinNet()
    model_ref = AsinNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arcsin():
    class ArcsinNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arcsin(x)

    model = ArcsinNet()
    model_ref = ArcsinNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_asinh():
    class AsinhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.asinh(x)

    model = AsinhNet()
    model_ref = AsinhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arcsinh():
    class ArcsinhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arcsinh(x)

    model = ArcsinhNet()
    model_ref = ArcsinhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_atan():
    class AtanNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.atan(x)

    model = AtanNet()
    model_ref = AtanNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arctan():
    class ArctanNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arctan(x)

    model = ArctanNet()
    model_ref = ArctanNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_atanh():
    class AtanhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.atanh(x)

    model = AtanhNet()
    model_ref = AtanhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arctanh():
    class ArctanhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arctanh(x)

    model = ArctanhNet()
    model_ref = ArctanhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_atan2():
    class Atan2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.atan2(x, x)

    model = Atan2Net()
    model_ref = Atan2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_arctan2():
    class Arctan2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.arctan2(x, x)

    model = Arctan2Net()
    model_ref = Arctan2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_bitwise_not():
    class BitwiseNotNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_not(x)

    model = BitwiseNotNet()
    model_ref = BitwiseNotNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_bitwise_and():
    class BitwiseAndNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_and(x, x)

    model = BitwiseAndNet()
    model_ref = BitwiseAndNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_bitwise_or():
    class BitwiseOrNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_or(x, x)

    model = BitwiseOrNet()
    model_ref = BitwiseOrNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_bitwise_xor():
    class BitwiseXorNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_xor(x, x)

    model = BitwiseXorNet()
    model_ref = BitwiseXorNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_bitwise_left_shift():
    class BitwiseLeftShiftNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_left_shift(x, x)

    model = BitwiseLeftShiftNet()
    model_ref = BitwiseLeftShiftNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Incorrect result")
def test_bitwise_right_shift():
    class BitwiseRightShiftNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.bitwise_right_shift(x, x)

    model = BitwiseRightShiftNet()
    model_ref = BitwiseRightShiftNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_ceil():
    class CeilNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ceil(x, min=-1, max=1)

    model = CeilNet()
    model_ref = CeilNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_clip():
    class ClipNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.clip(x, min=-1, max=1)

    model = ClipNet()
    model_ref = ClipNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_conj_physical():
    class ConjPhysicalNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.conj_physical(x)

    model = ConjPhysicalNet()
    model_ref = ConjPhysicalNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_copysign():
    class CopysignNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.copysign(x, 42)

    model = CopysignNet()
    model_ref = CopysignNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_cos():
    class CosNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.cos(x)

    model = CosNet()
    model_ref = CosNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_cosh():
    class CoshNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.cosh(x)

    model = CoshNet()
    model_ref = CoshNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_deg2rad():
    class Deg2RadNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.deg2rad(x)

    model = Deg2RadNet()
    model_ref = Deg2RadNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_div():
    class DivNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.div(x, x)

    model = DivNet()
    model_ref = DivNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_divide():
    class DivideNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.divide(x, x)

    model = DivideNet()
    model_ref = DivideNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_digamma():
    class DigammaNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.digamma(x)

    model = DigammaNet()
    model_ref = DigammaNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_erf():
    class ErfNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erf(x)

    model = ErfNet()
    model_ref = ErfNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_erfc():
    class ErfcNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erfc(x)

    model = ErfcNet()
    model_ref = ErfcNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_erfinv():
    class ErfinvNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erfinv(x)

    model = ErfinvNet()
    model_ref = ErfinvNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_exp():
    class ExpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp(x)

    model = ExpNet()
    model_ref = ExpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_exp2():
    class Exp2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp2(x)

    model = Exp2Net()
    model_ref = Exp2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_expm1():
    class Expm1Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.expm1(x)

    model = Expm1Net()
    model_ref = Expm1Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_fix():
    class FixNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.fix(x)

    model = FixNet()
    model_ref = FixNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_float_power():
    class FloatPowerNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.float_power(x, x)

    model = FloatPowerNet()
    model_ref = FloatPowerNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_floor():
    class FloorNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor(x)

    model = FloorNet()
    model_ref = FloorNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_floor_divide():
    class FloorDivideNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor_divide(x, x)

    model = FloorDivideNet()
    model_ref = FloorDivideNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_fmod():
    class FmodNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.fmod(x, x)

    model = FmodNet()
    model_ref = FmodNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_frac():
    class FracNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.frac(x)

    model = FracNet()
    model_ref = FracNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_frexp():
    class FrexpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.frexp(x)

    model = FrexpNet()
    model_ref = FrexpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_ldexp():
    class LdexpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ldexp(x, x)

    model = LdexpNet()
    model_ref = LdexpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_lerp():
    class LerpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.lerp(x, x, x)

    model = LerpNet()
    model_ref = LerpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_lgamma():
    class LgammaNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.lgamma(x)

    model = LgammaNet()
    model_ref = LgammaNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_log():
    class LogNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log(x)

    model = LogNet()
    model_ref = LogNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_log10():
    class Log10Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log10(x)

    model = Log10Net()
    model_ref = Log10Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_log1p():
    class Log1PNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log1p(x)

    model = Log1PNet()
    model_ref = Log1PNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_log2():
    class Log2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log2(x)

    model = Log2Net()
    model_ref = Log2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_logaddexp():
    class LogaddexpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logaddexp(x, x)

    model = LogaddexpNet()
    model_ref = LogaddexpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_logaddexp2():
    class Logaddexp2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logaddexp2(x, x)

    model = Logaddexp2Net()
    model_ref = Logaddexp2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_logical_and():
    class LogicalAndNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logical_and(x, x)

    model = LogicalAndNet()
    model_ref = LogicalAndNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_logical_not():
    class LogicalNotNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logical_not(x)

    model = LogicalNotNet()
    model_ref = LogicalNotNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_logical_or():
    class LogicalOrNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logical_or(x, x)

    model = LogicalOrNet()
    model_ref = LogicalOrNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_logical_xor():
    class LogicalXorNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logical_xor(x, x)

    model = LogicalXorNet()
    model_ref = LogicalXorNet()
    example_input = torch.tensor([-1, -2, 3], dtype=torch.int8)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_logit():
    class LogitNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.logit(x)

    model = LogitNet()
    model_ref = LogitNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_hypot():
    class HypotNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.hypot(x, x)

    model = HypotNet()
    model_ref = HypotNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_i0():
    class I0Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.i0(x)

    model = I0Net()
    model_ref = I0Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_igamma():
    class IgammaNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.igamma(x, x)

    model = IgammaNet()
    model_ref = IgammaNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_igammac():
    class IgammacNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.igammac(x, x)

    model = IgammacNet()
    model_ref = IgammacNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_mul():
    class MulNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mul(x, x)

    model = MulNet()
    model_ref = MulNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_multiply():
    class MultiplyNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.multiply(x, x)

    model = MultiplyNet()
    model_ref = MultiplyNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_mvlgamma():
    class MvlgammaNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mvlgamma(x, 2)

    model = MvlgammaNet()
    model_ref = MvlgammaNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_nan_to_num():
    class NanToNumNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.nan_to_num(x, nan=2.0, posinf=1.0)

    model = NanToNumNet()
    model_ref = NanToNumNet()
    example_input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_neg():
    class NegNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.neg(x)

    model = NegNet()
    model_ref = NegNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_negative():
    class NegativeNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.negative(x)

    model = NegativeNet()
    model_ref = NegativeNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_nextafter():
    class NextafterNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.nextafter(x, x)

    model = NextafterNet()
    model_ref = NextafterNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_positive():
    class PositiveNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.positive(x)

    model = PositiveNet()
    model_ref = PositiveNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_pow():
    class PowNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.pow(x, 2.0)

    model = PowNet()
    model_ref = PowNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_rad2deg():
    class Rad2DegNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.rad2deg(x)

    model = Rad2DegNet()
    model_ref = Rad2DegNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_real():
    class RealNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.real(x)

    model = RealNet()
    model_ref = RealNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_reciprocal():
    class ReciprocalNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.reciprocal(x)

    model = ReciprocalNet()
    model_ref = ReciprocalNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_remainder():
    class RemainderNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.remainder(x, x)

    model = RemainderNet()
    model_ref = RemainderNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_round():
    class RoundNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.round(x)

    model = RoundNet()
    model_ref = RoundNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_rsqrt():
    class RsqrtNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.rsqrt(x)

    model = RsqrtNet()
    model_ref = RsqrtNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_sigmoid():
    class SigmoidNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sigmoid(x)

    model = SigmoidNet()
    model_ref = SigmoidNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_sign():
    class SignNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sign(x)

    model = SignNet()
    model_ref = SignNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_sgn():
    class SgnNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sgn(x)

    model = SgnNet()
    model_ref = SgnNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_signbit():
    class SignbitNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.signbit(x)

    model = SignbitNet()
    model_ref = SignbitNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_sin():
    class SinNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sin(x)

    model = SinNet()
    model_ref = SinNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_sinc():
    class SincNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sinc(x)

    model = SincNet()
    model_ref = SincNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_sinh():
    class SinhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sinh(x)

    model = SinhNet()
    model_ref = SinhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_softmax():
    class SoftmaxNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.softmax(x, dim=1)

    model = SoftmaxNet()
    model_ref = SoftmaxNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_sqrt():
    class SqrtNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sqrt(x)

    model = SqrtNet()
    model_ref = SqrtNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_square():
    class SquareNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.square(x)

    model = SquareNet()
    model_ref = SquareNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

def test_sub():
    class SubNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sub(x, x)

    model = SubNet()
    model_ref = SubNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_subtract():
    class SubtractNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.subtract(x, x)

    model = SubtractNet()
    model_ref = SubtractNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_tan():
    class TanNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tan(x)

    model = TanNet()
    model_ref = TanNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires math dialect")
def test_tanh():
    class TanhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tanh(x)

    model = TanhNet()
    model_ref = TanhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_true_divide():
    class TrueDivideNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.true_divide(x, x)

    model = TrueDivideNet()
    model_ref = TrueDivideNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("MLIR frontend crahes")
def test_trunc():
    class TruncNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.trunc(x)

    model = TruncNet()
    model_ref = TruncNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)

@pytest.mark.skip("Unsupported by torch-mlir")
def test_xlogy():
    class XlogyNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.xlogy(x, x)

    model = XlogyNet()
    model_ref = XlogyNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)
    ref = model_ref(example_input)
    assert torch.allclose(res, ref)
