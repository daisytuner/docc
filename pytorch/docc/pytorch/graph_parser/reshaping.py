"""
GraphParser modules for parsing indexing, slicing, joining, and mutating operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, DebugInfo, Tensor

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class ConcatParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 1 or len(node.args) > 2:
            raise GraphParserError(
                self,
                node,
                "Expected between 1 and 2 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        if not isinstance(node.args[0], list):
            raise GraphParserError(
                self,
                node,
                "First argument must be a list type but got: "
                + str(type(node.args[0])),
            )
        num_args: int = len(node.args[0])
        tensor_infos: list[TensorInfo] = []
        for arg in node.args[0]:
            tensor_info: TensorInfo = self.convert_arg_to_tensor_info(
                node, metadata, arg
            )
            if not tensor_info.has_sdfg_tensor_type():
                raise GraphParserError(
                    self,
                    node,
                    "Expected an SDFG tensor type to be present: " + str(tensor_info),
                )
            if not tensor_info.has_container():
                raise GraphParserError(
                    self,
                    node,
                    "Expected an SDFG container to be present: " + str(tensor_info),
                )
            tensor_infos.append(tensor_info)

        if len(node.args) == 2:
            if not isinstance(node.args[1], int):
                raise GraphParserError(
                    self,
                    node,
                    "Second argument must be an int type but got: "
                    + str(type(node.args[1])),
                )
            dim: int = node.args[1]
        else:
            dim: int = 0
        if dim < 0:
            dim: int = dim + num_args

        result_info = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        builder.add_concat_op(
            [tensor_info.container() for tensor_info in tensor_infos],
            [tensor_info.sdfg_tensor_type() for tensor_info in tensor_infos],
            result_info.container(),
            result_info.sdfg_tensor_type(),
            dim,
            debug_info,
        )


register_module("aten.cat.default", ConcatParser())


class TensorReshape2dParser(GraphParserModule):
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

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        self.create_result_view(node, builder, metadata, self_info)


register_module("aten.permute.default", TensorReshape2dParser())
register_module("aten.squeeze.dims", TensorReshape2dParser())
register_module("aten.unsqueeze.default", TensorReshape2dParser())


class WhereParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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

        condition_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        other_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 2)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_conditional_copy_op(
            condition_info.container(),
            condition_info.sdfg_tensor_type(),
            self_info.container(),
            self_info.sdfg_tensor_type(),
            other_info.container(),
            other_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.where.self", WhereParser())


class SelectParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_tensor: Tensor | None = self.get_node_sdfg_tensor(node)
        if result_tensor is None:
            raise GraphParserError(self, node, "Unable to get result SDFG tensor")

        copy: bool = (
            metadata.has_tensor(node.name)
            and metadata.tensor(node.name).has_container()
            and metadata.has_container(metadata.tensor(node.name).container())
        )
        if copy:
            result_info: TensorInfo = self.get_result_tensor_info(
                node, builder, metadata
            )
            new_result_tensor: Tensor = Tensor(
                result_tensor.element_type, result_tensor.shape
            )
            debug_info: DebugInfo = self.get_debug_info(node)
            builder.add_copy_op(
                self_info.container(),
                result_tensor,
                result_info.container(),
                new_result_tensor,
                debug_info,
            )
            result_info.set_sdfg_tensor_type(new_result_tensor)
        else:
            self.create_result_view(node, builder, metadata, self_info)


register_module("aten.select.int", SelectParser())


class IndexParser(GraphParserModule):
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

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        indices: Argument = node.args[1]
        if not isinstance(indices, list):
            raise GraphParserError(
                self,
                node,
                "Expected indices arg to be list type but got: " + str(type(indices)),
            )
        index_positions: list[int] = []
        index_infos: list[TensorInfo] = []
        for i in range(len(indices)):
            index: Argument = indices[i]
            if not index is None:
                index_positions.append(i)
                index_infos.append(
                    self.convert_arg_to_tensor_info(node, metadata, index)
                )
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        builder.add_index_op(
            result_info.container(),
            result_info.sdfg_tensor_type(),
            self_info.container(),
            self_info.sdfg_tensor_type(),
            [index_info.container() for index_info in index_infos],
            [index_info.sdfg_tensor_type() for index_info in index_infos],
            index_positions,
            debug_info,
        )


register_module("aten.index.Tensor", IndexParser())


class ViewCopyParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) != 2:
            raise GraphParserError(
                self, node, "Expected exactly 2 argument but got " + str(len(node.args))
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        size: list[str] = self.get_arg_multi_expr(node, 1)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        if size != result_info.shape():
            raise GraphParserError(
                self, node, f"Shapes mismatch: {size} != {result_info.shape()}"
            )

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_copy_op(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.view_copy.default", ViewCopyParser())
