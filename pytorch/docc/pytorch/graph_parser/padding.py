"""
GraphParser modules for parsing padding layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserModule,
    GraphParserError,
    register_module,
)


class ConstandPadParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 2 or len(node.args) > 3:
            raise GraphParserError(
                self,
                node,
                "Expected between 2 and 3 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        pads: list[str] = self.get_arg_multi_expr(node, 1)
        if len(node.args) == 3:
            value_info_or_const: TensorInfo | TensorConstant = (
                self.get_arg_tensor_info_or_constant(
                    node, metadata, 2, align_constant_type=self_info.element_type()
                )
            )
        else:
            value_info_or_const: TensorInfo | TensorConstant = TensorConstant(
                "0", self_info.element_type()
            )
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        builder.add_const_padding_op(
            result_info.container(),
            result_info.sdfg_tensor_type(),
            self_info.container(),
            self_info.sdfg_tensor_type(),
            value_info_or_const.container(),
            value_info_or_const.element_type(),
            pads,
            debug_info,
        )


register_module("aten.constant_pad_nd.default", ConstandPadParser())
