"""
GraphParser modules for parsing non-linear activation functions.
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


class ReLUParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 1:
            raise GraphParserError(
                self,
                node,
                "Expected exactly one argument but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_relu(
            self_container, self_tensor, result_container, result_tensor, debug_info
        )


register_module("aten.relu.default", ReLUParser())


class GELUParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 1:
            raise GraphParserError(
                self,
                node,
                "Expected exactly one argument but got " + str(len(node.args)),
            )
        tanh_approx: bool = False
        if "approximate" in node.kwargs:
            approximate: Argument = node.kwargs["approximate"]
            if not isinstance(approximate, str):
                raise GraphParserError(
                    self,
                    node,
                    "Expected approximate kwarg to be str type but got: "
                    + str(type(approximate)),
                )
            if not approximate in ["none", "tanh"]:
                raise GraphParserError(
                    self, node, "Unknown approximation: " + approximate
                )
            if approximate == "tanh":
                tanh_approx: bool = True
        elif len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_gelu(
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            tanh_approx,
            debug_info,
        )


register_module("aten.gelu.default", GELUParser())


class SoftmaxParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 3:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 3 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        dim: Argument = node.args[1]
        if not isinstance(dim, int):
            raise GraphParserError(
                self, node, "Expected dim arg to be int type but got: " + str(type(dim))
            )
        half_to_float: Argument = node.args[2]
        if not isinstance(half_to_float, bool):
            raise GraphParserError(
                self,
                node,
                "Expected half_to_float arg to be bool type but got: "
                + str(type(half_to_float)),
            )
        if half_to_float:
            raise GraphParserError(
                self, node, "Currently setting half_to_float arg is unsupported"
            )
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_reduce_op(
            "softmax",
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            [dim],
            False,
            debug_info,
        )


register_module("aten._softmax.default", SoftmaxParser())
