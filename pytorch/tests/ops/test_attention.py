"""Tests for fused scaled-dot-product attention (AttentionNode).

Covers the parameter surface the node supports: batch/head leading dims, GQA/MQA
(K/V with fewer heads), causal masking, custom scale, and an additive float mask
(including broadcasting). Each case is compiled through the ``docc`` backend and
compared against PyTorch's own scaled_dot_product_attention.
"""

import pytest
import torch
import torch.nn.functional as F

from docc.sdfg import (
    ConstantNode,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFG,
    StructuredSDFGBuilder,
    Tensor,
)
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
    def __init__(self, scale=None) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)


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


@pytest.mark.parametrize("scale", [0.0, 0.5, -0.25, 0.12345678901234566])
def test_attention_custom_scale(target: str, scale: float) -> None:
    q, k, v = _qkv(1, 2, 8, 8)
    check(SDPANet(scale=scale), q, k, v, target=target, atol=1e-4)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("primitive_type", [PrimitiveType.Float, PrimitiveType.Double])
def test_attention_scale_constant(masked: bool, primitive_type) -> None:
    builder = StructuredSDFGBuilder("attention_scale_constant")
    scalar_type = Scalar(primitive_type)
    tensor_type = Tensor(scalar_type, ["4", "8"])
    for name in ("O", "Q", "K", "V"):
        builder.add_container(name, Pointer(scalar_type), is_argument=True)

    scale = 0.12345678901234566
    operands = ("O", tensor_type, "Q", tensor_type, "K", tensor_type, "V", tensor_type)
    if masked:
        builder.add_container("M", Pointer(scalar_type), is_argument=True)
        mask_type = Tensor(scalar_type, ["4", "4"])
        builder.add_attention_masked_op(*operands, "M", mask_type, scale, False)
    else:
        builder.add_attention_op(*operands, scale, False)

    sdfg = builder.move()
    for graph in (sdfg, StructuredSDFG.parse(sdfg.to_json())):
        graph.validate()
        edges = list(graph.root.child(0).dataflow.edges)
        assert len(edges) == (6 if masked else 5)
        scale_edges = [edge for edge in edges if edge.dst_conn == "scale"]
        assert len(scale_edges) == 1
        constant = scale_edges[0].src
        assert isinstance(constant, ConstantNode)
        assert constant.type.primitive_type == primitive_type
        assert float(constant.data) == scale


# --- additive mask (matching and broadcast) ---


@pytest.mark.parametrize("scale", [None, 0.12345678901234566])
def test_attention_additive_mask(target: str, scale) -> None:
    q, k, v = _qkv(1, 2, 8, 8)
    torch.manual_seed(1)
    mask = torch.randn(1, 2, 8, 8)
    check(SDPAMaskNet(scale=scale), q, k, v, mask, target=target, atol=1e-4)


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
