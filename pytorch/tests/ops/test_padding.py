import torch
import torch.nn as nn

from tests import check

# --- ZeroPad1d ---


def test_zero_pad_1d_simple(target: str) -> None:
    class ZeroPad1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_1d: nn.ZeroPad1d = nn.ZeroPad1d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_1d(input)

    check(ZeroPad1dSimpleNet(), torch.randn(2, 4), target=target)


def test_zero_pad_1d_tuple(target: str) -> None:
    class ZeroPad1dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_1d: nn.ZeroPad1d = nn.ZeroPad1d((3, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_1d(input)

    check(ZeroPad1dTupleNet(), torch.randn(2, 3), target=target)


def test_zero_pad_1d_batch(target: str) -> None:
    class ZeroPad1dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_1d: nn.ZeroPad1d = nn.ZeroPad1d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_1d(input)

    check(ZeroPad1dBatchNet(), torch.randn(1, 2, 4), target=target)


def test_zero_pad_1d_batch_tuple(target: str) -> None:
    class ZeroPad1dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_1d: nn.ZeroPad1d = nn.ZeroPad1d((3, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_1d(input)

    check(ZeroPad1dBatchTupleNet(), torch.randn(1, 2, 3), target=target)


# --- ZeroPad2d ---


def test_zero_pad_2d_simple(target: str) -> None:
    class ZeroPad2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_2d: nn.ZeroPad2d = nn.ZeroPad2d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_2d(input)

    check(ZeroPad2dSimpleNet(), torch.randn(1, 3, 3), target=target)


def test_zero_pad_2d_tuple(target: str) -> None:
    class ZeroPad2dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_2d: nn.ZeroPad2d = nn.ZeroPad2d((1, 1, 2, 0))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_2d(input)

    check(ZeroPad2dTupleNet(), torch.randn(1, 3, 3), target=target)


def test_zero_pad_2d_batch(target: str) -> None:
    class ZeroPad2dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_2d: nn.ZeroPad2d = nn.ZeroPad2d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_2d(input)

    check(ZeroPad2dBatchNet(), torch.randn(1, 1, 3, 3), target=target)


def test_zero_pad_2d_batch_tuple(target: str) -> None:
    class ZeroPad2dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_2d: nn.ZeroPad2d = nn.ZeroPad2d((1, 1, 2, 0))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_2d(input)

    check(ZeroPad2dBatchTupleNet(), torch.randn(1, 1, 3, 3), target=target)


# --- ZeroPad3d ---


def test_zero_pad_3d_simple(target: str) -> None:
    class ZeroPad3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_3d: nn.ZeroPad3d = nn.ZeroPad3d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_3d(input)

    check(ZeroPad3dSimpleNet(), torch.randn(3, 10, 20, 30), target=target)


def test_zero_pad_3d_tuple(target: str) -> None:
    class ZeroPad3dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_3d: nn.ZeroPad3d = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_3d(input)

    check(ZeroPad3dTupleNet(), torch.randn(3, 10, 20, 30), target=target)


def test_zero_pad_3d_batch(target: str) -> None:
    class ZeroPad3dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_3d: nn.ZeroPad3d = nn.ZeroPad3d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_3d(input)

    check(ZeroPad3dBatchNet(), torch.randn(16, 3, 10, 20, 30), target=target)


def test_zero_pad_3d_batch_tuple(target: str) -> None:
    class ZeroPad3dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.zero_pad_3d: nn.ZeroPad3d = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.zero_pad_3d(input)

    check(ZeroPad3dBatchTupleNet(), torch.randn(16, 3, 10, 20, 30), target=target)


# --- ConstantPad1d ---


def test_constant_pad_1d_simple(target: str) -> None:
    class ConstantPad1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_1d: nn.ConstantPad1d = nn.ConstantPad1d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_1d(input)

    check(ConstantPad1dSimpleNet(), torch.randn(2, 4), target=target)


def test_constant_pad_1d_tuple(target: str) -> None:
    class ConstantPad1dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_1d: nn.ConstantPad1d = nn.ConstantPad1d((3, 1), 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_1d(input)

    check(ConstantPad1dTupleNet(), torch.randn(2, 3), target=target)


def test_constant_pad_1d_batch(target: str) -> None:
    class ConstantPad1dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_1d: nn.ConstantPad1d = nn.ConstantPad1d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_1d(input)

    check(ConstantPad1dBatchNet(), torch.randn(1, 2, 4), target=target)


def test_constant_pad_1d_batch_tuple(target: str) -> None:
    class ConstantPad1dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_1d: nn.ConstantPad1d = nn.ConstantPad1d((3, 1), 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_1d(input)

    check(ConstantPad1dBatchTupleNet(), torch.randn(1, 2, 3), target=target)


# --- ConstantPad2d ---


def test_constant_pad_2d_simple(target: str) -> None:
    class ConstantPad2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_2d: nn.ConstantPad2d = nn.ConstantPad2d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_2d(input)

    check(ConstantPad2dSimpleNet(), torch.randn(1, 2, 2), target=target)


def test_constant_pad_2d_tuple(target: str) -> None:
    class ConstantPad2dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_2d: nn.ConstantPad2d = nn.ConstantPad2d((3, 0, 2, 1), 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_2d(input)

    check(ConstantPad2dTupleNet(), torch.randn(1, 2, 2), target=target)


def test_constant_pad_2d_batch(target: str) -> None:
    class ConstantPad2dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_2d: nn.ConstantPad2d = nn.ConstantPad2d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_2d(input)

    check(ConstantPad2dBatchNet(), torch.randn(3, 1, 2, 2), target=target)


def test_constant_pad_2d_batch_tuple(target: str) -> None:
    class ConstantPad2dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_2d: nn.ConstantPad2d = nn.ConstantPad2d((3, 0, 2, 1), 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_2d(input)

    check(ConstantPad2dBatchTupleNet(), torch.randn(3, 1, 2, 2), target=target)


# --- ConstantPad3d ---


def test_constant_pad_3d_simple(target: str) -> None:
    class ConstantPad3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_3d: nn.ConstantPad3d = nn.ConstantPad3d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_3d(input)

    check(ConstantPad3dSimpleNet(), torch.randn(3, 10, 20, 30), target=target)


def test_constant_pad_3d_tuple(target: str) -> None:
    class ConstantPad3dTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_3d: nn.ConstantPad3d = nn.ConstantPad3d(
                (3, 3, 6, 6, 0, 1), 3.5
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_3d(input)

    check(ConstantPad3dTupleNet(), torch.randn(3, 10, 20, 30), target=target)


def test_constant_pad_3d_batch(target: str) -> None:
    class ConstantPad3dBatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_3d: nn.ConstantPad3d = nn.ConstantPad3d(2, 3.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_3d(input)

    check(ConstantPad3dBatchNet(), torch.randn(16, 3, 10, 20, 30), target=target)


def test_constant_pad_3d_batch_tuple(target: str) -> None:
    class ConstantPad3dBatchTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.constant_pad_3d: nn.ConstantPad3d = nn.ConstantPad3d(
                (3, 3, 6, 6, 0, 1), 3.5
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.constant_pad_3d(input)

    check(ConstantPad3dBatchTupleNet(), torch.randn(16, 3, 10, 20, 30), target=target)
