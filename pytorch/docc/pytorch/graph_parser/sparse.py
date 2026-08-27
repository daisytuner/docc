"""
GraphParser modules for parsing sparse operations.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class EmbeddingParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        # aten.embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
        # The padding_idx, scale_grad_by_freq and sparse arguments only affect the
        # backward pass and are therefore ignored for this forward-only lowering.
        if len(node.args) < 2:
            raise GraphParserError(
                self,
                node,
                "Expected at least 2 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        weight_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        if len(weight_info.shape()) != 2:
            raise GraphParserError(
                self,
                node,
                "Embedding weight must be 2-dimensional but got shape: "
                + str(weight_info.shape()),
            )
        index_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_embedding_op(
            weight_info.container(),
            weight_info.sdfg_tensor_type(),
            index_info.container(),
            index_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.embedding.default", EmbeddingParser())


class EmbeddingRenormParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 4 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        if len(self_info.shape()) != 2:
            raise GraphParserError(
                self,
                node,
                "Weight must be 2-dimensional but got shape: " + str(self_info.shape()),
            )
        indices_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        max_norm_constant: TensorConstant = self.get_arg_tensor_constant(
            node, 2, align_constant_type=self_info.element_type()
        )
        norm_type_constant: TensorConstant = self.get_arg_tensor_constant(
            node, 3, align_constant_type=self_info.element_type()
        )
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_embedding_renorm_op(
            result_info.container(),
            result_info.sdfg_tensor_type(),
            self_info.container(),
            self_info.sdfg_tensor_type(),
            indices_info.container(),
            indices_info.sdfg_tensor_type(),
            max_norm_constant.value(),
            max_norm_constant.sdfg_scalar(),
            norm_type_constant.value(),
            norm_type_constant.sdfg_scalar(),
            debug_info,
        )


register_module("aten.embedding_renorm.default", EmbeddingRenormParser())
