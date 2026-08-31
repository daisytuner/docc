import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check

# --- pad ---


def test_pad_simple(target: str) -> None:
    class PadSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.pad(input, (1, 1), "constant", 0)

    check(PadSimpleNet(), torch.randn(3, 3, 4, 2), target=target)


def test_pad_constant_full(target: str) -> None:
    class PadFullNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.pad(input, (1, 1, 1, 1, 1, 1, 1, 1), "constant", 0)

    check(PadFullNet(), torch.randn(3, 3, 4, 2), target=target)


# --- interpolate ---


def test_upsample_bilinear_functional_size(target: str) -> None:
    class UpsampleBilinearFunctionalSizeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                input, size=(28, 28), mode="bilinear", align_corners=False
            )

    check(UpsampleBilinearFunctionalSizeNet(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_functional_scale(target: str) -> None:
    class UpsampleBilinearFunctionalScaleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                input, scale_factor=2.0, mode="bilinear", align_corners=True
            )

    check(
        UpsampleBilinearFunctionalScaleNet(), torch.randn(2, 3, 16, 16), target=target
    )
