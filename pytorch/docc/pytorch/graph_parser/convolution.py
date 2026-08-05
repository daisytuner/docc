"""
GraphParser modules for parsing convolution layers.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo, Type

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
        # DOCC's convolution expansion always writes a contiguous (NCHW) output.
        # When torch.compile selects a non-contiguous memory format for the
        # convolution (e.g. channels_last), the fx metadata expresses the result
        # -- and every downstream view/permute derived from it -- relative to that
        # layout. Emit the convolution into a contiguous buffer and copy it into
        # the requested (strided) layout so the physical data matches what the
        # consumers assume; otherwise the layout mismatch silently transposes the
        # data (observed as a wrong LayerNorm result in SegFormer).
        conv_container: str = result_container
        conv_tensor: Tensor = result_tensor
        relayout: bool = not result_tensor.is_contiguous()
        if relayout:
            result_type: Type = container_info[result_container].sdfg_type()
            conv_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
            conv_container = self.create_intermediate_container(
                node, builder, container_info, result_type, conv_tensor
            )
        if node.args[2] is None:
            builder.add_conv(
                input_container,
                input_tensor,
                weight_container,
                weight_tensor,
                conv_container,
                conv_tensor,
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
                conv_container,
                conv_tensor,
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
        if relayout:
            builder.add_copy_op(
                conv_container,
                conv_tensor,
                result_container,
                result_tensor,
                debug_info,
            )


register_module("aten.convolution.default", ConvolutionParser())
