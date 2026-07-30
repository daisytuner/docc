"""
GraphParser modules for parsing indexing, slicing, joining, and mutating operations.
"""

import torch.fx

from docc.sdfg import StructuredSDFGBuilder, Tensor, DebugInfo

from docc.pytorch.graph_parser.utils import (
    ContainerInfoBase,
    ContainerPreInfo,
    ContainerInfos,
    GraphParserError,
    GraphParserModule,
    register_pre_module,
    register_module,
)


class ConcatParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        tensor_containers: list[str] = []
        tensor_tensors: list[Tensor] = []
        for arg in node.args[0]:
            tensor_container: str = self.convert_arg_to_container(
                node, container_info, arg
            )
            tensor_containers.append(tensor_container)
            tensor_tensors.append(
                self.get_tensor_type(node, container_info, tensor_container)
            )
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
        result_container = self.get_result_container(node, builder, container_info)
        result_tensor = self.get_tensor_type(node, container_info, result_container)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_concat_op(
            tensor_containers,
            tensor_tensors,
            result_container,
            result_tensor,
            dim,
            debug_info,
        )


register_module("aten.cat.default", ConcatParser())


class SliceParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) < 1 or len(node.args) > 5:
            raise GraphParserError(
                self,
                node,
                "Expected between 1 and 5 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        rank: int = len(self_tensor.shape)

        dim: int = node.args[1] if len(node.args) > 1 else 0
        if not isinstance(dim, int):
            raise GraphParserError(
                self, node, "Expected dim arg to be int type but got: " + str(type(dim))
            )
        if dim < 0:
            dim = dim + rank
        if dim < 0 or dim >= rank:
            raise GraphParserError(self, node, "Slice dim out of range: " + str(dim))

        try:
            size: int = int(self_tensor.shape[dim])
        except (TypeError, ValueError):
            raise GraphParserError(
                self,
                node,
                "Slice requires a static size along dim "
                + str(dim)
                + " but got: "
                + str(self_tensor.shape[dim]),
            )

        start: object = node.args[2] if len(node.args) > 2 else None
        end: object = node.args[3] if len(node.args) > 3 else None
        step: int = node.args[4] if len(node.args) > 4 else 1

        if not isinstance(step, int):
            raise GraphParserError(
                self,
                node,
                "Expected step arg to be int type but got: " + str(type(step)),
            )
        if step <= 0:
            raise GraphParserError(
                self, node, "Only positive step is supported but got: " + str(step)
            )

        if start is None:
            start = 0
        elif not isinstance(start, int):
            raise GraphParserError(
                self,
                node,
                "Expected start arg to be int type but got: " + str(type(start)),
            )
        elif start < 0:
            start = start + size
        start = max(0, min(start, size))

        if end is None:
            end = size
        elif not isinstance(end, int):
            raise GraphParserError(
                self,
                node,
                "Expected end arg to be int type but got: " + str(type(end)),
            )
        elif end < 0:
            end = end + size
        end = max(start, min(end, size))

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        # PyTorch models a slice as a view, so its tensor metadata carries the
        # parent's (non-contiguous) strides. This library node materializes a
        # contiguous copy, so we register the result with C-strides to keep the
        # allocation, the write and any downstream reads consistent.
        result_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        container_info[result_container].update(sdfg_tensor_type=result_tensor)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_slice_op(
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            dim,
            start,
            end,
            step,
            debug_info,
        )


register_module("aten.slice.Tensor", SliceParser())
register_module("aten.slice_copy.Tensor", SliceParser())


class TensorReshape2dParser(GraphParserModule):
    def pre_parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        container: str = node.name
        ref_container: str = self.get_arg_container(
            node, container_info, 0, resolve=False
        )
        if container in container_info:
            info: ContainerInfoBase = container_info[container]
            if not isinstance(info, ContainerPreInfo):
                raise GraphParserError(
                    self, node, "Expected ContainerPreInfo but got: " + str(type(info))
                )
            container_info[container] = ContainerPreInfo.copy(info, ref=ref_container)
        else:
            container_info[container] = ContainerPreInfo(container, ref=ref_container)
        if ref_container in container_info:
            info: ContainerInfoBase = container_info[ref_container]
            if not isinstance(info, ContainerPreInfo):
                raise GraphParserError(
                    self, node, "Expected ContainerPreInfo but got: " + str(type(info))
                )
            container_info[ref_container] = ContainerPreInfo.copy(
                info, refed_by=container
            )
        else:
            container_info[ref_container] = ContainerPreInfo(
                ref_container, refed_by=container
            )

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        self.update_container_types(node, builder, container_info, node.name)


register_pre_module("aten.permute.default", TensorReshape2dParser())
register_module("aten.permute.default", TensorReshape2dParser())
register_pre_module("aten.squeeze.dims", TensorReshape2dParser())
register_module("aten.squeeze.dims", TensorReshape2dParser())
register_pre_module("aten.unsqueeze.default", TensorReshape2dParser())
register_module("aten.unsqueeze.default", TensorReshape2dParser())


class EmbeddingParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        # aten.embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
        # The padding_idx, scale_grad_by_freq and sparse arguments only affect the
        # backward pass and are therefore ignored for this forward-only lowering.
        if len(node.args) < 2:
            raise GraphParserError(
                self,
                node,
                "Expected at least 2 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        weight_container: str = self.get_arg_container(node, container_info, 0)
        weight_tensor: Tensor = self.get_tensor_type(
            node, container_info, weight_container
        )
        if len(weight_tensor.shape) != 2:
            raise GraphParserError(
                self,
                node,
                "Embedding weight must be 2-dimensional but got shape: "
                + str(weight_tensor.shape),
            )
        index_container: str = self.get_arg_container(node, container_info, 1)
        index_tensor: Tensor = self.get_tensor_type(
            node, container_info, index_container
        )

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        # `embedding` produces a fresh contiguous tensor. Register the result with
        # C-strides so the allocation, the write and any downstream reads agree.
        result_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        container_info[result_container].update(sdfg_tensor_type=result_tensor)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_embedding_op(
            weight_container,
            weight_tensor,
            index_container,
            index_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.embedding.default", EmbeddingParser())


def _register_alias(
    module: GraphParserModule,
    node: torch.fx.Node,
    container_info: ContainerInfos,
    ref_container: str,
) -> None:
    """Registers ``node.name`` as an alias (reference) of ``ref_container``."""
    container: str = node.name
    if container in container_info:
        info: ContainerInfoBase = container_info[container]
        if not isinstance(info, ContainerPreInfo):
            raise GraphParserError(
                module, node, "Expected ContainerPreInfo but got: " + str(type(info))
            )
        container_info[container] = ContainerPreInfo.copy(info, ref=ref_container)
    else:
        container_info[container] = ContainerPreInfo(container, ref=ref_container)
    if ref_container in container_info:
        info = container_info[ref_container]
        if not isinstance(info, ContainerPreInfo):
            raise GraphParserError(
                module, node, "Expected ContainerPreInfo but got: " + str(type(info))
            )
        container_info[ref_container] = ContainerPreInfo.copy(info, refed_by=container)
    else:
        container_info[ref_container] = ContainerPreInfo(
            ref_container, refed_by=container
        )


class EmbeddingRenormParser(GraphParserModule):
    def pre_parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        # embedding_renorm_ mutates `self` in place and returns it. Alias the
        # result name to `self` so any downstream reference resolves to the same
        # (renormalized) container.
        if len(node.args) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 4 arguments but got " + str(len(node.args)),
            )
        ref_container: str = self.get_arg_container(
            node, container_info, 0, resolve=False
        )
        _register_alias(self, node, container_info, ref_container)

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        # aten.embedding_renorm_(self, indices, max_norm, norm_type) -> self
        if len(node.args) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 4 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        if len(self_tensor.shape) != 2:
            raise GraphParserError(
                self,
                node,
                "embedding_renorm_ weight must be 2-dimensional but got shape: "
                + str(self_tensor.shape),
            )
        index_container: str = self.get_arg_container(node, container_info, 1)
        index_tensor: Tensor = self.get_tensor_type(
            node, container_info, index_container
        )
        max_norm_arg = node.args[2]
        norm_type_arg = node.args[3]
        if not isinstance(max_norm_arg, (int, float)):
            raise GraphParserError(
                self,
                node,
                "max_norm must be a number but got: " + str(type(max_norm_arg)),
            )
        if not isinstance(norm_type_arg, (int, float)):
            raise GraphParserError(
                self,
                node,
                "norm_type must be a number but got: " + str(type(norm_type_arg)),
            )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_embedding_renorm_op(
            self_container,
            self_tensor,
            index_container,
            index_tensor,
            float(max_norm_arg),
            float(norm_type_arg),
            debug_info,
        )
        self.update_container_types(node, builder, container_info, node.name)


register_pre_module("aten.embedding_renorm.default", EmbeddingRenormParser())
register_module("aten.embedding_renorm.default", EmbeddingRenormParser())


class IndexParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        rank: int = len(self_tensor.shape)

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

        index_containers: list[str] = []
        index_tensors: list[Tensor] = []
        for i in index_positions:
            index_container: str = self.convert_arg_to_container(
                node, container_info, indices[i]
            )
            index_tensor: Tensor = self.get_tensor_type(
                node, container_info, index_container
            )
            index_containers.append(index_container)
            index_tensors.append(index_tensor)

        # All index tensors must broadcast to a common shape; only identically
        # shaped index tensors are supported here.
        reference_shape = index_tensors[0].shape
        for index_tensor in index_tensors[1:]:
            if index_tensor.shape != reference_shape:
                raise GraphParserError(
                    self,
                    node,
                    "Only identically shaped index tensors are supported but got: "
                    + str([t.shape for t in index_tensors]),
                )

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        # `index.Tensor` produces a fresh contiguous tensor. Register the result
        # with C-strides so the allocation, the write and any downstream reads agree.
        result_tensor = Tensor(result_tensor.element_type, result_tensor.shape)
        container_info[result_container].update(sdfg_tensor_type=result_tensor)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_index_op(
            self_container,
            self_tensor,
            index_containers,
            index_tensors,
            result_container,
            result_tensor,
            dim_offset,
            debug_info,
        )


register_module("aten.index.Tensor", IndexParser())
