"""
GraphParser modules for parsing pooling layers.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    ContainerInfos,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class MaxPoolWithIndicesParser(GraphParserModule):
    _dim: int

    def __init__(self, dim: int) -> None:
        self._dim: int = dim

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) < 3 or len(node.args) > 6:
            raise GraphParserError(
                self,
                node,
                "Expected between 3 and 6 arguments but got " + str(len(node.args)),
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        kernel_size: list[str] = self.get_arg_multi_expr(node, 1)
        if len(kernel_size) != self._dim:
            raise GraphParserError(
                self, node, f"Expected {self._dim}d kernel_size but got: {kernel_size}"
            )
        stride: list[str] = self.get_arg_multi_expr(node, 2)
        if len(stride) != self._dim:
            raise GraphParserError(
                self, node, f"Expected {self._dim}d stride but got: {stride}"
            )
        if len(node.args) >= 4:
            padding: list[str] = self.get_arg_multi_expr(node, 3)
            if len(padding) != self._dim:
                raise GraphParserError(
                    self, node, f"Expected {self._dim}d padding but got: {padding}"
                )
        else:
            padding: list[str] = ["0" for _ in range(self._dim)]
        padding: list[str] = padding + padding
        if len(node.args) >= 5:
            dilation: list[str] = self.get_arg_multi_expr(node, 4)
            if len(dilation) != self._dim:
                raise GraphParserError(
                    self, node, f"Expected {self._dim}d dilation but got: {dilation}"
                )
        else:
            dilation: list[str] = ["1" for _ in range(self._dim)]
        if len(node.args) == 6:
            arg_5: Argument = node.args[5]
            if not isinstance(arg_5, bool):
                raise GraphParserError(
                    self,
                    node,
                    "Expected bool as 6th argument but got: " + str(type(node.args[5])),
                )
            ceil_mode: bool = arg_5
        else:
            ceil_mode: bool = False
        if ceil_mode:
            raise GraphParserError(self, node, "ceil_mode == True is not supported")
        result_containers: tuple[str, ...] = self.get_result_containers(
            2, node, builder, container_info
        )
        result_container: str = result_containers[0]
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_pooling(
            "max",
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            self_tensor.shape,
            kernel_size,
            stride,
            padding,
            dilation,
            debug_info,
        )


register_module("aten.max_pool2d_with_indices.default", MaxPoolWithIndicesParser(2))
register_module("aten.max_pool3d_with_indices.default", MaxPoolWithIndicesParser(3))
