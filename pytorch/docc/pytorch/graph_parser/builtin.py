"""
GraphParser modules for parsing builtin Python functions.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder

from docc.pytorch.graph_parser.utils import (
    TensorName,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class GetitemParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 2:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 2 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        first: Argument = node.args[0]
        second: Argument = node.args[1]
        if not isinstance(first, torch.fx.Node):
            raise GraphParserError(
                self,
                node,
                "Expected first argument to be forch.fx.Node type but got: "
                + str(type(first)),
            )
        if not isinstance(second, int):
            raise GraphParserError(
                self,
                node,
                "Expected second argument to be int type but got: " + str(type(second)),
            )

        ref_name: TensorName = f"{first.name}_{second}"
        if not metadata.has_tensor(ref_name):
            raise GraphParserError(
                self, node, f"Could not find tensor information for '{ref_name}'"
            )
        self.create_result_view(node, builder, metadata, metadata.tensor(ref_name))


register_module("_operator.getitem", GetitemParser())
