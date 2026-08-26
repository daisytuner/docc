import torch
import torch.nn as nn

from tests import check

# --- LayerNorm on non-contiguous (permuted) inputs ---


def test_view_permute_layernorm(target: str) -> None:
    class ViewPermuteLayerNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = input.view(2, 3, 5)
            input = torch.permute(input, (2, 0, 1))
            return self.layernorm(input)

    check(ViewPermuteLayerNormNet().eval(), torch.randn(2, 3, 5), target=target)


def test_reshape_permute_layernorm(target: str) -> None:
    # Mirrors SegFormer's efficient self-attention sequence-reduction path:
    # a 4D feature map is flattened and transposed so that the channel dim
    # (which is normalized over) is the trailing, non-unit-stride axis.
    class ReshapePermuteLayerNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(32)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            batch_size, num_channels, height, width = input.shape
            input = input.reshape(batch_size, num_channels, height * width)
            input = input.permute(0, 2, 1)
            return self.layernorm(input)

    check(
        ReshapePermuteLayerNormNet().eval(), torch.randn(1, 32, 16, 16), target=target
    )


def test_transpose_layernorm(target: str) -> None:
    class TransposeLayerNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(32)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = input.transpose(1, 2)
            return self.layernorm(input)

    check(TransposeLayerNormNet().eval(), torch.randn(1, 32, 256), target=target)
