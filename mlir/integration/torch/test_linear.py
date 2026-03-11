import torch
import torch.nn as nn
import pytest

import docc.torch


# --- Single linear layer (no bias) ---


def test_single_nobias_compile():
    class SingleNoBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleNoBiasNet(10, 5)
    model_ref = SingleNoBiasNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_single_nobias_backend():
    class SingleNoBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleNoBiasNet(10, 5)
    model_ref = SingleNoBiasNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Single linear layer (with bias) ---


def test_single_bias_compile():
    class SingleBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleBiasNet(10, 5)
    model_ref = SingleBiasNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_single_bias_backend():
    class SingleBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleBiasNet(10, 5)
    model_ref = SingleBiasNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Chained linear layers (no bias) ---


def test_chained_nobias_compile():
    class ChainedNoBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = ChainedNoBiasNet(10, 16, 3)
    model_ref = ChainedNoBiasNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_chained_nobias_backend():
    class ChainedNoBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = ChainedNoBiasNet(10, 16, 3)
    model_ref = ChainedNoBiasNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Chained linear layers (with bias) ---


def test_chained_bias_compile():
    class ChainedBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = ChainedBiasNet(10, 16, 3)
    model_ref = ChainedBiasNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_chained_bias_backend():
    class ChainedBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = ChainedBiasNet(10, 16, 3)
    model_ref = ChainedBiasNet(10, 16, 3)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Non-square dimensions ---


def test_wide_output_compile():
    class WideOutputNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = WideOutputNet(4, 32)
    model_ref = WideOutputNet(4, 32)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_wide_output_backend():
    class WideOutputNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = WideOutputNet(4, 32)
    model_ref = WideOutputNet(4, 32)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Narrow bottleneck ---


def test_bottleneck_compile():
    class BottleneckNet(nn.Module):
        def __init__(self, in_features: int, bottleneck: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, bottleneck, bias=False)
            self.linear2 = nn.Linear(bottleneck, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = BottleneckNet(32, 2, 32)
    model_ref = BottleneckNet(32, 2, 32)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_bottleneck_backend():
    class BottleneckNet(nn.Module):
        def __init__(self, in_features: int, bottleneck: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, bottleneck, bias=False)
            self.linear2 = nn.Linear(bottleneck, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    model = BottleneckNet(32, 2, 32)
    model_ref = BottleneckNet(32, 2, 32)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Single sample (batch_size=1) ---


def test_single_sample_compile():
    class SingleSampleNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleSampleNet(10, 5)
    model_ref = SingleSampleNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_single_sample_backend():
    class SingleSampleNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SingleSampleNet(10, 5)
    model_ref = SingleSampleNet(10, 5)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Deep linear stack (3 layers, no bias) ---


def test_deep_stack_compile():
    class DeepStackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 16, bias=False)
            self.linear2 = nn.Linear(16, 8, bias=False)
            self.linear3 = nn.Linear(8, 3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear3(self.linear2(self.linear1(x)))

    model = DeepStackNet()
    model_ref = DeepStackNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_deep_stack_backend():
    class DeepStackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 16, bias=False)
            self.linear2 = nn.Linear(16, 8, bias=False)
            self.linear3 = nn.Linear(8, 3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear3(self.linear2(self.linear1(x)))

    model = DeepStackNet()
    model_ref = DeepStackNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Square linear layer (in_features == out_features) ---


def test_square_compile():
    class SquareLinearNet(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.linear = nn.Linear(features, features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SquareLinearNet(16)
    model_ref = SquareLinearNet(16)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 16)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_square_backend():
    class SquareLinearNet(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.linear = nn.Linear(features, features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = SquareLinearNet(16)
    model_ref = SquareLinearNet(16)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 16)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


# --- Scalar output (out_features=1) ---


def test_scalar_output_compile():
    class ScalarOutputNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, 1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = ScalarOutputNet(10)
    model_ref = ScalarOutputNet(10)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_scalar_output_backend():
    class ScalarOutputNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, 1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = ScalarOutputNet(10)
    model_ref = ScalarOutputNet(10)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(8, 10)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-5)
