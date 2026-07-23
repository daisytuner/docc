"""
GraphParser modules for parsing non-linear activation functions.
"""

import torch.fx

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
