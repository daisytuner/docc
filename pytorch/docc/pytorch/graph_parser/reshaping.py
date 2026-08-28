"""
GraphParser modules for parsing indexing, slicing, joining, and mutating operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

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
        rank: int = len(self_info.shape())

        if not isinstance(node.args[1], (list, tuple)):
            raise GraphParserError(
                self,
                node,
                "Second argument must be a list of indices but got: "
                + str(type(node.args[1])),
            )
        indices: list = list(node.args[1])
        if len(indices) > rank:
            raise GraphParserError(
                self,
                node,
                "Got more indices ("
                + str(len(indices))
                + ") than tensor rank ("
                + str(rank)
                + ")",
            )

        # Collect the positions of the advanced (non-None) index tensors. Only a
        # single contiguous block of index tensors is supported.
        index_positions: list[int] = [
            i for i, idx in enumerate(indices) if idx is not None
        ]
        if len(index_positions) == 0:
            raise GraphParserError(
                self,
                node,
                "Expected at least one index tensor but got only None entries",
            )
        dim_offset: int = index_positions[0]
        num_indices: int = len(index_positions)
        if index_positions != list(range(dim_offset, dim_offset + num_indices)):
            raise GraphParserError(
                self,
                node,
                "Only a contiguous block of index tensors is supported but got positions: "
                + str(index_positions),
            )

        index_infos: list[TensorInfo] = []
        for i in index_positions:
            index_infos.append(
                self.convert_arg_to_tensor_info(node, metadata, indices[i])
            )

        # All index tensors must broadcast to a common shape; only identically
        # shaped index tensors are supported here.
        reference_shape = index_infos[0].shape()
        for index_info in index_infos[1:]:
            if index_info.shape() != reference_shape:
                raise GraphParserError(
                    self,
                    node,
                    "Only identically shaped index tensors are supported but got: "
                    + str([t.shape() for t in index_infos]),
                )

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_index_op(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            [index_info.container() for index_info in index_infos],
            [index_info.sdfg_tensor_type() for index_info in index_infos],
            result_info.container(),
            result_info.sdfg_tensor_type(),
            dim_offset,
            debug_info,
        )


register_module("aten.index.Tensor", IndexParser())
