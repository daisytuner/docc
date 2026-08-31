import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check

# --- Upsample by explicit output size, align_corners=False ---


def test_upsample_bilinear_size_upscale(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_downscale(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_asymmetric(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(24, 12), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_non_integer_ratio(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(23, 19), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 4, 16, 16), target=target)


def test_upsample_bilinear_size_identity(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Upsample by explicit output size, align_corners=True ---


def test_upsample_bilinear_size_upscale_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_downscale_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_asymmetric_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(30, 10), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_non_integer_ratio_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(21, 27), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 4, 16, 16), target=target)


# --- Upsample by scale factor, align_corners=False ---


def test_upsample_bilinear_scale_2x(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=2.0, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_half(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=0.5, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_non_integer(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=1.5, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_asymmetric(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=(2.0, 3.0), mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Upsample by scale factor, align_corners=True ---


def test_upsample_bilinear_scale_2x_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_non_integer_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_asymmetric_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=(3.0, 2.0), mode="bilinear", align_corners=True
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Different input shapes (batch / channels / rectangular inputs) ---


def test_upsample_bilinear_single_batch_single_channel(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(20, 20), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 1, 8, 8), target=target)


def test_upsample_bilinear_many_channels(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=2.0, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 32, 8, 8), target=target)


def test_upsample_bilinear_rectangular_input(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(16, 40), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 8, 20), target=target)


def test_upsample_bilinear_rectangular_input_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(40, 16), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 20, 8), target=target)
