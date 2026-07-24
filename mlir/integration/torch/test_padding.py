import pytest

import torch
import torch.nn as nn

from integration.torch.check import check_backend


@pytest.mark.skip("Unsupported by torch-mlir")
def test_reflection_pad_1d():
    class ReflectionPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad1d((3, 1))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(
        ReflectionPad1dNet().eval(), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
    )


@pytest.mark.skip("Unsupported by torch-mlir")
def test_reflection_pad_2d():
    class ReflectionPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad2d((1, 1, 2, 0))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(
        ReflectionPad2dNet().eval(),
        torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3),
    )


@pytest.mark.skip("Unsupported by torch-mlir")
def test_reflection_pad_3d():
    class ReflectionPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad3d(1)

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(
        ReflectionPad3dNet().eval(),
        torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2),
    )


def test_replication_pad_1d():
    class ReplicationPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad1d((3, 1))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(
        ReplicationPad1dNet().eval(),
        torch.arange(8, dtype=torch.float).reshape(1, 2, 4),
    )


def test_replication_pad_2d():
    class ReplicationPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad2d((1, 1, 2, 0))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(
        ReplicationPad2dNet().eval(),
        torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3),
    )


def test_replication_pad_3d():
    class ReplicationPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ReplicationPad3dNet().eval(), torch.randn(16, 3, 8, 320, 480))


def test_zero_pad_1d():
    class ZeroPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad1d((3, 1))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ZeroPad1dNet().eval(), torch.randn(1, 2, 4))


def test_zero_pad_2d():
    class ZeroPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad2d((1, 1, 2, 0))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ZeroPad2dNet().eval(), torch.randn(1, 1, 3, 3))


def test_zero_pad_3d():
    class ZeroPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ZeroPad3dNet().eval(), torch.randn(16, 3, 10, 20, 30))


def test_constant_pad_1d():
    class ConstantPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((3, 1), 3.5)

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ConstantPad1dNet().eval(), torch.randn(1, 2, 4))


def test_constant_pad_2d():
    class ConstantPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad2d((3, 0, 2, 1), 3.5)

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ConstantPad2dNet().eval(), torch.randn(1, 2, 2))


def test_constant_pad_3d():
    class ConstantPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)

        def forward(self, x: torch.Tensor):
            return self.pad(x)

    check_backend(ConstantPad3dNet().eval(), torch.randn(16, 3, 10, 20, 30))


# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_1d():
#     class CircularPad1dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad1d((3, 1))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     check_backend(CircularPad1dNet().eval(), torch.arange(8, dtype=torch.float).reshape(1, 2, 4))

# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_2d():
#     class CircularPad2dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad2d((1, 1, 2, 0))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     check_backend(CircularPad2dNet().eval(), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3))

# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_3d():
#     class CircularPad3dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     check_backend(CircularPad3dNet().eval(), torch.randn(16, 3, 8, 320, 480))
