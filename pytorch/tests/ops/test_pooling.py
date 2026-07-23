import torch
import torch.nn as nn

from tests import check

# --- MaxPool1d ---


def test_maxpool1d_simple(target: str) -> None:
    class MaxPool1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool1d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool1dSimpleNet(), torch.randn(2, 3, 16), target=target)


def test_maxpool1d_stride(target: str) -> None:
    class MaxPool1dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool1d(2, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool1dStrideNet(), torch.randn(2, 3, 16), target=target)


def test_maxpool1d_padding(target: str) -> None:
    class MaxPool1dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool1d(4, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool1dPaddingNet(), torch.randn(2, 3, 16), target=target)


def test_maxpool1d_dilation(target: str) -> None:
    class MaxPool1dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool1d(2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool1dDilationNet(), torch.randn(2, 3, 16), target=target)


def test_maxpool1d_complex(target: str) -> None:
    class MaxPool1dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool1d(4, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool1dComplexNet(), torch.randn(2, 3, 16), target=target)


# --- MaxPool2d ---


def test_maxpool2d_simple(target: str) -> None:
    class MaxPool2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dSimpleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_tuple_kernel(target: str) -> None:
    class MaxPool2dTupleKernelNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d((2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dTupleKernelNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_stride(target: str) -> None:
    class MaxPool2dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(2, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dStrideNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_tuple_stride(target: str) -> None:
    class MaxPool2dTupleStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(2, stride=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dTupleStrideNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_padding(target: str) -> None:
    class MaxPool2dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(4, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dPaddingNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_tuple_padding(target: str) -> None:
    class MaxPool2dTuplePaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(4, padding=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dTuplePaddingNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_dilation(target: str) -> None:
    class MaxPool2dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dDilationNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_tuple_dilation(target: str) -> None:
    class MaxPool2dTupleDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(2, dilation=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dTupleDilationNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_complex(target: str) -> None:
    class MaxPool2dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(4, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dComplexNet(), torch.randn(2, 3, 16, 16), target=target)


def test_maxpool2d_tuple_complex(target: str) -> None:
    class MaxPool2dTupleComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool2d(
                (4, 2), stride=(2, 1), padding=(2, 1), dilation=(2, 1)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool2dTupleComplexNet(), torch.randn(2, 3, 16, 16), target=target)


# --- MaxPool3d ---


def test_maxpool3d_simple(target: str) -> None:
    class MaxPool3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dSimpleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_tuple_kernel(target: str) -> None:
    class MaxPool3dTupleKernelNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d((3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dTupleKernelNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_stride(target: str) -> None:
    class MaxPool3dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(2, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dStrideNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_tuple_stride(target: str) -> None:
    class MaxPool3dTupleStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(2, stride=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dTupleStrideNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_padding(target: str) -> None:
    class MaxPool3dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(6, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dPaddingNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_tuple_padding(target: str) -> None:
    class MaxPool3dTuplePaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(6, padding=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dTuplePaddingNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_dilation(target: str) -> None:
    class MaxPool3dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dDilationNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_tuple_dilation(target: str) -> None:
    class MaxPool3dTupleDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(2, dilation=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dTupleDilationNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_complex(target: str) -> None:
    class MaxPool3dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(4, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dComplexNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_maxpool3d_tuple_complex(target: str) -> None:
    class MaxPool3dTupleComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maxpool = nn.MaxPool3d(
                (6, 4, 2), stride=(3, 2, 1), padding=(3, 2, 1), dilation=(3, 2, 1)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.maxpool(input)

    check(MaxPool3dTupleComplexNet(), torch.randn(2, 3, 16, 16, 16), target=target)
