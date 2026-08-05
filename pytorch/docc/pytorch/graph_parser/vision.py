"""
GraphParser modules for parsing predefined vision operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    ContainerInfoBase,
    ContainerInfo,
    ContainerRefInfo,
    ContainerPreInfo,
    ContainerInfos,
    GraphParserError,
    GraphParserModule,
    register_pre_module,
    register_module,
)


class UpsampleBilinear2DParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) < 3 or len(node.args) > 4:
            raise GraphParserError(
                self,
                node,
                "Expected 3 or 4 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        input_container: str = self.get_arg_container(node, container_info, 0)
        input_tensor: Tensor = self.get_tensor_type(
            node, container_info, input_container
        )
        if len(input_tensor.shape) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected 4D input [N, C, H, W] but got: " + str(input_tensor.shape),
            )
        align_corners_arg: Argument = node.args[2]
        if not isinstance(align_corners_arg, bool):
            raise GraphParserError(
                self,
                node,
                "Expected bool align_corners but got: " + str(type(align_corners_arg)),
            )
        align_corners: bool = align_corners_arg
        scale_factors: list[float] = []
        if len(node.args) == 3 and node.args[2] is None:
            raise GraphParserError(
                self,
                node,
                "Scale factors cannot be None if sizes are not specified: "
                + str(node.args[2]),
            )
        elif len(node.args) == 3 and node.args[1] is None:
            raise GraphParserError(
                self,
                node,
                "Sizes cannot be None if scale factors are not specified: "
                + str(node.args[1]),
            )
        elif (
            len(node.args) == 4
            and node.args[1] is not None
            and node.args[3] is not None
        ):
            raise GraphParserError(
                self,
                node,
                "Expected either sizes or scale factors but got both: "
                + str(node.args[1])
                + " and "
                + str(node.args[3]),
            )

        if len(node.args) == 4 and node.args[3] is not None:
            if not isinstance(node.args[3], (list)):
                raise GraphParserError(
                    self,
                    node,
                    "Expected list of scale factors but got: "
                    + str(type(node.args[3])),
                )
            scale_factors = []
            for scale_factor in node.args[3]:
                if not isinstance(scale_factor, (float, int)):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected float scale factor but got: "
                        + str(type(scale_factor)),
                    )
                scale_factors.append(float(scale_factor))
            if len(scale_factors) != 2:
                raise GraphParserError(
                    self,
                    node,
                    "Expected 2 scale factors but got: " + str(scale_factors),
                )
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        if len(result_tensor.shape) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected 4D output [N, C, H, W] but got: " + str(result_tensor.shape),
            )
        # interpolate preserves its input's memory format, so the fx metadata can
        # express the result in a channels_last layout. Graph output arguments are
        # boundary tensors that are always contiguous (NCHW); the channels_last
        # metadata does not apply to them. Writing the strided layout into the
        # output boundary silently transposes the data, so force a contiguous
        # write when the result is a graph output.
        is_output: bool = container_info[result_container].out_argument()
        if is_output and not result_tensor.is_contiguous():
            result_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_upsample_bilinear2d(
            input_container,
            input_tensor,
            result_container,
            result_tensor,
            input_tensor.shape,
            result_tensor.shape,
            align_corners,
            scale_factors,
            debug_info,
        )


register_module("aten.upsample_bilinear2d.vec", UpsampleBilinear2DParser())
