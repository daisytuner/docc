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

        # DOCC's convolution expansion always writes a contiguous (NCHW) output.
        # When torch.compile selects a non-contiguous memory format for the
        # convolution (e.g. channels_last), the fx metadata expresses the result
        # -- and every downstream view/permute derived from it -- relative to that
        # layout. Emit the convolution into a contiguous buffer and copy it into
        # the requested (strided) layout so the physical data matches what the
        # consumers assume; otherwise the layout mismatch silently transposes the
        # data.
        # Graph output arguments are boundary tensors and always contiguous
        # (NCHW); the fx channels_last metadata does not apply to them, so never
        # relayout into it -- doing so transposes the physical data. The
        # expansion's GEMM already writes the output contiguously, so the output
        # tensor must be described as contiguous as well; otherwise the bias add
        # would address the boundary buffer in the channels_last layout and
        # disagree with the GEMM.
        # TODO: Reintegrate!
        # conv_container: str = result_container
        # conv_tensor: Tensor = result_tensor
        # is_output: bool = container_info[result_container].out_argument()
        # if is_output and not result_tensor.is_contiguous():
        #     conv_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        # relayout: bool = not result_tensor.is_contiguous() and not is_output
        # if relayout:
        #     result_type: Type = container_info[result_container].sdfg_type()
        #     conv_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        #     conv_container = self.create_intermediate_container(
        #         node, builder, container_info, result_type, conv_tensor
        #     )

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
        # if relayout:
        #     builder.add_copy_op(
        #         conv_container,
        #         conv_tensor,
        #         result_container,
        #         result_tensor,
        #         debug_info,
        #     )


register_module("aten.convolution.default", ConvolutionParser())
