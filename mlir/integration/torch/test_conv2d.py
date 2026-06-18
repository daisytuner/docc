import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile

# --- Single Conv2d (no bias) ---


def test_single_nobias_compile():
    class SingleConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(SingleConv2dNet().eval(), torch.randn(1, 3, 32, 32), rtol=3e-4)


def test_single_nobias_backend():
    class SingleConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(SingleConv2dNet().eval(), torch.randn(1, 3, 32, 32), rtol=3e-4)


# --- Single Conv2d (with bias) ---


def test_single_bias_compile():
    class SingleConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=True)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(SingleConv2dBiasNet().eval(), torch.randn(1, 3, 32, 32))


def test_single_bias_backend():
    class SingleConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=True)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(SingleConv2dBiasNet().eval(), torch.randn(1, 3, 32, 32))


# --- Chained Conv2d layers ---


def test_chained_compile():
    class ChainedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv2(self.conv1(x))

    check_compile(ChainedConv2dNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-5)


def test_chained_backend():
    class ChainedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv2(self.conv1(x))

    check_backend(ChainedConv2dNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-5)


# --- Different kernel sizes ---


def test_kernel_1x1_compile():
    class Conv1x1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(Conv1x1Net().eval(), torch.randn(1, 3, 32, 32), rtol=1e-04)


def test_kernel_1x1_backend():
    class Conv1x1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(Conv1x1Net().eval(), torch.randn(1, 3, 32, 32), rtol=1e-04)


def test_kernel_5x5_compile():
    class Conv5x5Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(Conv5x5Net().eval(), torch.randn(1, 3, 32, 32), atol=1e-06)


def test_kernel_5x5_backend():
    class Conv5x5Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(Conv5x5Net().eval(), torch.randn(1, 3, 32, 32), atol=1e-06)


# --- Padding ---


def test_padding_compile():
    class PaddedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(PaddedConv2dNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-06)


def test_padding_backend():
    class PaddedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(PaddedConv2dNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-06)


# --- Stride > 1 ---


def test_stride_compile():
    class StridedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(StridedConv2dNet(), torch.randn(1, 3, 32, 32), atol=1e-06)


def test_stride_backend():
    class StridedConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(StridedConv2dNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-06)


# --- Batch size > 1 ---


def test_batch_compile():
    class BatchConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(BatchConv2dNet().eval(), torch.randn(4, 3, 32, 32), atol=1e-06)


def test_batch_backend():
    class BatchConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(BatchConv2dNet().eval(), torch.randn(4, 3, 32, 32), atol=1e-06)


# --- Single output channel ---


def test_single_channel_out_compile():
    class SingleChannelOutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(SingleChannelOutNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-07)


def test_single_channel_out_backend():
    class SingleChannelOutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(SingleChannelOutNet().eval(), torch.randn(1, 3, 32, 32), atol=1e-07)


# --- Depthwise Conv2d (groups=in_channels) ---


def test_depthwise_compile():
    class DepthwiseConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(DepthwiseConv2dNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_backend():
    class DepthwiseConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(DepthwiseConv2dNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_bias_compile():
    class DepthwiseConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(DepthwiseConv2dBiasNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_bias_backend():
    class DepthwiseConv2dBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, groups=16)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(DepthwiseConv2dBiasNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_padding_compile():
    class DepthwiseConv2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                16, 16, kernel_size=3, padding=1, groups=16, bias=False
            )

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(DepthwiseConv2dPaddingNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_padding_backend():
    class DepthwiseConv2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                16, 16, kernel_size=3, padding=1, groups=16, bias=False
            )

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(DepthwiseConv2dPaddingNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_padding_bias_compile():
    class DepthwiseConv2dPaddingBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_compile(DepthwiseConv2dPaddingBiasNet().eval(), torch.randn(1, 16, 32, 32))


def test_depthwise_padding_bias_backend():
    class DepthwiseConv2dPaddingBiasNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    check_backend(DepthwiseConv2dPaddingBiasNet().eval(), torch.randn(1, 16, 32, 32))
