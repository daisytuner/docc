import torch
import torch.nn as nn
import pytest

import docc.torch

# --- Single Conv2d (no bias) ---


def test_single_nobias_compile():
    class SingleConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleConv2dNet()
    model_ref = SingleConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=3e-4)


def test_single_nobias_backend():
    class SingleConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleConv2dNet()
    model_ref = SingleConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=3e-4)


# --- Single Conv2d (with bias) ---


def test_single_bias_compile():
    class SingleConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=True)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleConv2dBiasNet()
    model_ref = SingleConv2dBiasNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_single_bias_backend():
    class SingleConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=True)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleConv2dBiasNet()
    model_ref = SingleConv2dBiasNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


# --- Chained Conv2d layers ---


def test_chained_compile():
    class ChainedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv2(self.conv1(x))

    model = ChainedConv2dNet()
    model_ref = ChainedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-5)


def test_chained_backend():
    class ChainedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv2(self.conv1(x))

    model = ChainedConv2dNet()
    model_ref = ChainedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-5)


# --- Different kernel sizes ---


def test_kernel_1x1_compile():
    class Conv1x1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = Conv1x1Net()
    model_ref = Conv1x1Net()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-04)


def test_kernel_1x1_backend():
    class Conv1x1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = Conv1x1Net()
    model_ref = Conv1x1Net()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-04)


def test_kernel_5x5_compile():
    class Conv5x5Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = Conv5x5Net()
    model_ref = Conv5x5Net()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


def test_kernel_5x5_backend():
    class Conv5x5Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = Conv5x5Net()
    model_ref = Conv5x5Net()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


# --- Padding ---


def test_padding_compile():
    class PaddedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = PaddedConv2dNet()
    model_ref = PaddedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


def test_padding_backend():
    class PaddedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = PaddedConv2dNet()
    model_ref = PaddedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


# --- Stride > 1 ---


def test_stride_compile():
    class StridedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = StridedConv2dNet()
    model_ref = StridedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


def test_stride_backend():
    class StridedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = StridedConv2dNet()
    model_ref = StridedConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


# --- Batch size > 1 ---


def test_batch_compile():
    class BatchConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = BatchConv2dNet()
    model_ref = BatchConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(4, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


def test_batch_backend():
    class BatchConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = BatchConv2dNet()
    model_ref = BatchConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(4, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-06)


# --- Single output channel ---


def test_single_channel_out_compile():
    class SingleChannelOutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleChannelOutNet()
    model_ref = SingleChannelOutNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-07)


def test_single_channel_out_backend():
    class SingleChannelOutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleChannelOutNet()
    model_ref = SingleChannelOutNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, atol=1e-07)


# --- Depthwise Conv2d (groups=in_channels) ---


def test_depthwise_compile():
    class DepthwiseConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = DepthwiseConv2dNet()
    model_ref = DepthwiseConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 16, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_depthwise_backend():
    class DepthwiseConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = DepthwiseConv2dNet()
    model_ref = DepthwiseConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 16, 32, 32)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)
