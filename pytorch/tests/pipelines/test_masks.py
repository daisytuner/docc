import torch
import torch.nn as nn
from transformers.masking_utils import (
    sdpa_mask,
    sliding_window_causal_mask_function,
    chunked_causal_mask_function,
)

import pytest

from tests import check

# --- sdpa_mask ---


def test_sdpa_mask_simple(target: str) -> None:
    class SdpaMaskSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor | None:
            return sdpa_mask(
                batch_size=1,
                q_length=5,
                kv_length=5,
                allow_is_causal_skip=False,
            )

    check(SdpaMaskSimpleNet(), *(), target=target)


@pytest.mark.skip(reason="Needs support for aten.sub.Tensor")
def test_sdpa_mask_sliding_window(target: str) -> None:
    class SdpaMaskSlidingWindowNet(nn.Module):
        def forward(self) -> torch.Tensor | None:
            return sdpa_mask(
                batch_size=1,
                q_length=5,
                kv_length=5,
                mask_function=sliding_window_causal_mask_function(3),
                allow_is_causal_skip=False,
            )

    check(SdpaMaskSlidingWindowNet(), *(), target=target)


@pytest.mark.skip(reason="Needs support for aten.sub.Tensor")
def test_sdpa_mask_chunked(target: str) -> None:
    class SdpaMaskChunkedNet(nn.Module):
        def forward(self) -> torch.Tensor | None:
            return sdpa_mask(
                batch_size=1,
                q_length=5,
                kv_length=5,
                mask_function=chunked_causal_mask_function(
                    3, torch.zeros(1, dtype=torch.int)
                ),
                allow_is_causal_skip=False,
            )

    check(SdpaMaskChunkedNet(), *(), target=target)


def test_sdpa_mask_qwen(target: str) -> None:
    class SdpaMaskQwenNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor | None:
            return sdpa_mask(
                batch_size=1,
                q_length=39,
                kv_length=39,
                attention_mask=input,
                allow_is_causal_skip=False,
            )

    check(SdpaMaskQwenNet(), torch.full((1, 39), True), target=target)
