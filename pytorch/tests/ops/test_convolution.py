import torch
import torch.nn as nn

from tests import check

# --- Conv1d ---


def test_conv1d_simple(target: str) -> None:
    class Conv1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dSimpleNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_bias(target: str) -> None:
    class Conv1dBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_stride(target: str) -> None:
    class Conv1dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, stride=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dStrideNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_stride_bias(target: str) -> None:
    class Conv1dStrideBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dStrideBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_padding(target: str) -> None:
    class Conv1dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, padding=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dPaddingNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_padding_bias(target: str) -> None:
    class Conv1dPaddingBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dPaddingBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_dilation(target: str) -> None:
    class Conv1dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, dilation=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dDilationNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_dilation_bias(target: str) -> None:
    class Conv1dDilationBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dDilationBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_groups(target: str) -> None:
    class Conv1dGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, groups=3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dGroupsNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_groups_bias(target: str) -> None:
    class Conv1dGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, groups=3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dGroupsBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_complex(target: str) -> None:
    class Conv1dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(
                3, 6, 3, stride=2, padding=2, dilation=2, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dComplexNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_complex_bias(target: str) -> None:
    class Conv1dComplexBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(3, 6, 3, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dComplexBiasNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_complex_groups(target: str) -> None:
    class Conv1dComplexGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dComplexGroupsNet(), torch.randn(2, 3, 16), target=target)


def test_conv1d_complex_groups_bias(target: str) -> None:
    class Conv1dComplexGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d: nn.Conv1d = nn.Conv1d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv1d(input)

    check(Conv1dComplexGroupsBiasNet(), torch.randn(2, 3, 16), target=target)


# --- Conv2d ---


def test_conv2d_simple(target: str) -> None:
    class Conv2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dSimpleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_bias(target: str) -> None:
    class Conv2dBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_stride(target: str) -> None:
    class Conv2dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, stride=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dStrideNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_stride_bias(target: str) -> None:
    class Conv2dStrideBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dStrideBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_stride_tuple(target: str) -> None:
    class Conv2dStrideTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, stride=(2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dStrideTupleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_stride_tuple_bias(target: str) -> None:
    class Conv2dStrideTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, stride=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dStrideTupleBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_padding(target: str) -> None:
    class Conv2dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, padding=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dPaddingNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_padding_bias(target: str) -> None:
    class Conv2dPaddingBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dPaddingBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_padding_tuple(target: str) -> None:
    class Conv2dPaddingTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, padding=(2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dPaddingTupleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_padding_tuple_bias(target: str) -> None:
    class Conv2dPaddingTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, padding=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dPaddingTupleBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_dilation(target: str) -> None:
    class Conv2dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, dilation=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dDilationNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_dilation_bias(target: str) -> None:
    class Conv2dDilationBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dDilationBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_dilation_tuple(target: str) -> None:
    class Conv2dDilationTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, dilation=(2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dDilationTupleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_dilation_tuple_bias(target: str) -> None:
    class Conv2dDilationTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, dilation=(2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dDilationTupleBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_groups(target: str) -> None:
    class Conv2dGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, groups=3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dGroupsNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_groups_bias(target: str) -> None:
    class Conv2dGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, groups=3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dGroupsBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex(target: str) -> None:
    class Conv2dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(
                3, 6, 3, stride=2, padding=2, dilation=2, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex_bias(target: str) -> None:
    class Conv2dComplexBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(3, 6, 3, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex_tuple(target: str) -> None:
    class Conv2dComplexTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(
                3, 6, 3, stride=(2, 1), padding=(2, 1), dilation=(2, 1), bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexTupleNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex_tuple_bias(target: str) -> None:
    class Conv2dComplexTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(
                3, 6, 3, stride=(2, 1), padding=(2, 1), dilation=(2, 1)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexTupleBiasNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex_groups(target: str) -> None:
    class Conv2dComplexGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexGroupsNet(), torch.randn(2, 3, 16, 16), target=target)


def test_conv2d_complex_groups_bias(target: str) -> None:
    class Conv2dComplexGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv2d: nn.Conv2d = nn.Conv2d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv2d(input)

    check(Conv2dComplexGroupsBiasNet(), torch.randn(2, 3, 16, 16), target=target)


# --- Conv3d ---


def test_conv3d_simple(target: str) -> None:
    class Conv3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dSimpleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_bias(target: str) -> None:
    class Conv3dBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_stride(target: str) -> None:
    class Conv3dStrideNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, stride=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dStrideNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_stride_bias(target: str) -> None:
    class Conv3dStrideBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, stride=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dStrideBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_stride_tuple(target: str) -> None:
    class Conv3dStrideTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, stride=(3, 2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dStrideTupleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_stride_tuple_bias(target: str) -> None:
    class Conv3dStrideTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, stride=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dStrideTupleBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_padding(target: str) -> None:
    class Conv3dPaddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, padding=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dPaddingNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_padding_bias(target: str) -> None:
    class Conv3dPaddingBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, padding=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dPaddingBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_padding_tuple(target: str) -> None:
    class Conv3dPaddingTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, padding=(3, 2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dPaddingTupleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_padding_tuple_bias(target: str) -> None:
    class Conv3dPaddingTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, padding=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dPaddingTupleBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_dilation(target: str) -> None:
    class Conv3dDilationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, dilation=2, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dDilationNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_dilation_bias(target: str) -> None:
    class Conv3dDilationBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dDilationBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_dilation_tuple(target: str) -> None:
    class Conv3dDilationTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, dilation=(3, 2, 1), bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dDilationTupleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_dilation_tuple_bias(target: str) -> None:
    class Conv3dDilationTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, dilation=(3, 2, 1))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dDilationTupleBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_groups(target: str) -> None:
    class Conv3dGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, groups=3, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dGroupsNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_groups_bias(target: str) -> None:
    class Conv3dGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, groups=3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dGroupsBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex(target: str) -> None:
    class Conv3dComplexNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(
                3, 6, 3, stride=2, padding=2, dilation=2, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex_bias(target: str) -> None:
    class Conv3dComplexBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(3, 6, 3, stride=2, padding=2, dilation=2)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex_tuple(target: str) -> None:
    class Conv3dComplexTupleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(
                3,
                6,
                3,
                stride=(3, 2, 1),
                padding=(3, 2, 1),
                dilation=(3, 2, 1),
                bias=False,
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexTupleNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex_tuple_bias(target: str) -> None:
    class Conv3dComplexTupleBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(
                3, 6, 3, stride=(3, 2, 1), padding=(3, 2, 1), dilation=(3, 2, 1)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexTupleBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex_groups(target: str) -> None:
    class Conv3dComplexGroupsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexGroupsNet(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_conv3d_complex_groups_bias(target: str) -> None:
    class Conv3dComplexGroupsBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3d: nn.Conv3d = nn.Conv3d(
                3, 6, 3, stride=2, padding=2, dilation=2, groups=3
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.conv3d(input)

    check(Conv3dComplexGroupsBiasNet(), torch.randn(2, 3, 16, 16, 16), target=target)
