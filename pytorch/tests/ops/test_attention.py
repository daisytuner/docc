"""Tests for fused scaled-dot-product attention (AttentionNode).

Covers the parameter surface the node supports: batch/head leading dims, GQA/MQA
(K/V with fewer heads), causal masking, custom scale, and an additive float mask
(including broadcasting). Each case is compiled through the ``docc`` backend and
compared against PyTorch's own scaled_dot_product_attention.
"""

import pytest
import torch
import torch.nn.functional as F

from tests import check


class SDPANet(torch.nn.Module):
    def __init__(
        self, is_causal: bool = False, scale=None, enable_gqa: bool = False
    ) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale
        self.enable_gqa = enable_gqa

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        kwargs = {}
        if self.scale is not None:
            kwargs["scale"] = self.scale
        if self.enable_gqa:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal, **kwargs
        )


class SDPAMaskNet(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _qkv(*shape: int):
    torch.manual_seed(0)
    return torch.randn(*shape), torch.randn(*shape), torch.randn(*shape)


# --- leading dims (single head, multi head, batch) ---


def test_attention_2d(target: str) -> None:
    q, k, v = _qkv(4, 8)
    check(SDPANet(), q, k, v, target=target, atol=1e-4)


def test_attention_single_batch_head(target: str) -> None:
    q, k, v = _qkv(1, 1, 8, 16)
    check(SDPANet(), q, k, v, target=target, atol=1e-4)


def test_attention_multihead(target: str) -> None:
    q, k, v = _qkv(2, 4, 16, 8)
    check(SDPANet(), q, k, v, target=target, atol=1e-4)


# --- causal ---


def test_attention_causal(target: str) -> None:
    q, k, v = _qkv(2, 2, 8, 8)
    check(SDPANet(is_causal=True), q, k, v, target=target, atol=1e-4)


# --- custom scale ---


def test_attention_custom_scale(target: str) -> None:
    q, k, v = _qkv(1, 2, 8, 8)
    check(SDPANet(scale=0.5), q, k, v, target=target, atol=1e-4)


# --- additive mask (matching and broadcast) ---


def test_attention_additive_mask(target: str) -> None:
    q, k, v = _qkv(1, 2, 8, 8)
    torch.manual_seed(1)
    mask = torch.randn(1, 2, 8, 8)
    check(SDPAMaskNet(), q, k, v, mask, target=target, atol=1e-4)


def test_attention_additive_mask_broadcast(target: str) -> None:
    q, k, v = _qkv(2, 4, 8, 8)
    torch.manual_seed(1)
    mask = torch.randn(1, 1, 8, 8)  # broadcast over batch and heads
    check(SDPAMaskNet(), q, k, v, mask, target=target, atol=1e-4)


# --- grouped-query / multi-query attention (K/V with fewer heads) ---


@pytest.mark.minimum_pytorch_version((2, 5, 0))
def test_attention_gqa(target: str) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 8)
    k = torch.randn(2, 2, 16, 8)
    v = torch.randn(2, 2, 16, 8)
    check(SDPANet(enable_gqa=True), q, k, v, target=target, atol=1e-4)


@pytest.mark.minimum_pytorch_version((2, 5, 0))
def test_attention_mqa(target: str) -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8, 8)
    k = torch.randn(1, 1, 8, 8)
    v = torch.randn(1, 1, 8, 8)
    check(SDPANet(enable_gqa=True), q, k, v, target=target, atol=1e-4)
