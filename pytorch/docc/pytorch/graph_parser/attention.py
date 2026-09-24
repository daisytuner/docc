"""
GraphParser module for scaled-dot-product attention.

Intercepts the single composite ``aten.scaled_dot_product_attention`` op (kept intact
by dropping it from the export decomposition table) and lowers it ourselves, rather than
letting PyTorch decompose it into bmm + softmax + bmm. For now it composes the explicit
math from existing tensor ops (Q Kᵀ → scale → softmax → · V); this is the interception
point where a fused ``AttentionNode`` will later replace the composition.
"""

import math

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    primitive_type_is_floating_point,
    register_module,
)


class ScaledDotProductAttentionParser(GraphParserModule):
    """Lowers ``aten.scaled_dot_product_attention.default(Q, K, V, ...)``."""

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 3:
            raise GraphParserError(
                self, node, "Expected at least Q, K, V but got " + str(len(node.args))
            )

        # Attributes: mask is supported (additive float); dropout must be off (inference).
        attn_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
        dropout_p = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        )
        is_causal = (
            node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        )
        if dropout_p:
            raise GraphParserError(self, node, "dropout_p != 0 is not supported yet")

        q_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        k_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        v_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 2)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        q_type: Tensor = q_info.sdfg_tensor_type()
        k_type: Tensor = k_info.sdfg_tensor_type()
        if len(q_type.shape) < 2:
            raise GraphParserError(
                self,
                node,
                "Q must be at least 2D ([..., N, D]) but got " + str(q_type.shape),
            )

        d = q_type.shape[-1]
        scale = node.kwargs.get("scale")
        if scale is None and len(node.args) > 6:
            scale = node.args[6]
        if scale is None:
            scale = 1.0 / math.sqrt(int(d))

        # O = softmax(scale · Q Kᵀ [+ mask]) V, lowered as one fused AttentionNode.
        if attn_mask is not None:
            mask_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 3)
            mask_type: Tensor = mask_info.sdfg_tensor_type()
            if not primitive_type_is_floating_point(
                mask_type.element_type.primitive_type
            ):
                raise GraphParserError(
                    self,
                    node,
                    "Only additive floating-point attn_mask is supported yet",
                )
            builder.add_attention_masked_op(
                result_info.container(),
                result_info.sdfg_tensor_type(),
                q_info.container(),
                q_type,
                k_info.container(),
                k_type,
                v_info.container(),
                v_info.sdfg_tensor_type(),
                mask_info.container(),
                mask_type,
                float(scale),
                bool(is_causal),
                debug_info,
            )
        else:
            builder.add_attention_op(
                result_info.container(),
                result_info.sdfg_tensor_type(),
                q_info.container(),
                q_type,
                k_info.container(),
                k_type,
                v_info.container(),
                v_info.sdfg_tensor_type(),
                float(scale),
                bool(is_causal),
                debug_info,
            )


register_module(
    "aten.scaled_dot_product_attention.default", ScaledDotProductAttentionParser()
)
