"""
GraphParser modules for parsing convolution layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, DebugInfo

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorMetadata,
    GraphParserModule,
    GraphParserError,
    register_module,
)


class ConvolutionParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 9:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 9 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        if not isinstance(node.args[6], bool) or node.args[6]:
            raise GraphParserError(
                self, node, "Currently only non-transposed convolutions are supported"
            )
        input_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        weight_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        stride: list[str] = self.get_arg_multi_expr(node, 3)
        padding: list[str] = self.get_arg_multi_expr(node, 4)
        padding_extended: list[str] = padding + padding
        dilation: list[str] = self.get_arg_multi_expr(node, 5)
        output_padding: list[str] = self.get_arg_multi_expr(node, 7)
        for pad in output_padding:
            if pad != "0":
                raise GraphParserError(
                    self,
                    node,
                    "Output padding for non-transposed convolution must be zero but got: "
                    + pad,
                )
        groups: str = self.get_arg_expr(node, 8)

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        if node.args[2] is None:
            builder.add_conv(
                input_info.container(),
                input_info.sdfg_tensor_type(),
                weight_info.container(),
                weight_info.sdfg_tensor_type(),
                result_info.container(),
                result_info.sdfg_tensor_type(),
                input_info.shape(),
                weight_info.shape()[2:],
                stride,
                padding_extended,
                dilation,
                weight_info.shape()[0],
                groups,
                debug_info,
            )
        else:
            bias_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 2)
            builder.add_conv_with_bias(
                input_info.container(),
                input_info.sdfg_tensor_type(),
                weight_info.container(),
                weight_info.sdfg_tensor_type(),
                result_info.container(),
                result_info.sdfg_tensor_type(),
                bias_info.container(),
                bias_info.sdfg_tensor_type(),
                input_info.shape(),
                weight_info.shape()[2:],
                stride,
                padding_extended,
                dilation,
                weight_info.shape()[0],
                groups,
                debug_info,
            )


register_module("aten.convolution.default", ConvolutionParser())
