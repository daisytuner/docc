"""
GraphParser modules for parsing convolution layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    GraphParserModule,
    ContainerInfos,
    GraphParserError,
    register_module,
)


class ConvolutionParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        input_container: str = self.get_arg_container(node, container_info, 0)
        input_tensor: Tensor = self.get_tensor_type(
            node, container_info, input_container
        )
        weight_container: str = self.get_arg_container(node, container_info, 1)
        weight_tensor: Tensor = self.get_tensor_type(
            node, container_info, weight_container
        )
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
        result_container = self.get_result_container(node, builder, container_info)
        result_tensor = self.get_tensor_type(node, container_info, result_container)
        debug_info: DebugInfo = self.get_debug_info(node)
        if node.args[2] is None:
            builder.add_conv(
                input_container,
                input_tensor,
                weight_container,
                weight_tensor,
                result_container,
                result_tensor,
                input_tensor.shape,
                weight_tensor.shape[2:],
                stride,
                padding_extended,
                dilation,
                weight_tensor.shape[0],
                groups,
                debug_info,
            )
        else:
            bias_container: str = self.get_arg_container(node, container_info, 2)
            bias_tensor: Tensor = self.get_tensor_type(
                node, container_info, bias_container
            )
            builder.add_conv_with_bias(
                input_container,
                input_tensor,
                weight_container,
                weight_tensor,
                result_container,
                result_tensor,
                bias_container,
                bias_tensor,
                input_tensor.shape,
                weight_tensor.shape[2:],
                stride,
                padding_extended,
                dilation,
                weight_tensor.shape[0],
                groups,
                debug_info,
            )


register_module("aten.convolution.default", ConvolutionParser())
