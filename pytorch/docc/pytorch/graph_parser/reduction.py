"""
GraphParser modules for parsing reduction operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import (
    AccessNode,
    Block,
    StructuredSDFGBuilder,
    Tasklet,
    TaskletCode,
    Tensor,
    Scalar,
    DebugInfo,
    Type,
    Pointer,
    PrimitiveType,
)

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class MinMaxParser(GraphParserModule):
    _op_type: str

    def __init__(self, op_type: str) -> None:
        super().__init__()
        self._op_type: str = op_type

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 1 or len(node.args) > 3:
            raise GraphParserError(
                self,
                node,
                "Expected between 1 and 3 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)

        axes: list[int] = [i for i in range(len(self_info.shape()))]
        if len(node.args) >= 2:
            dim: Argument = node.args[1]
            if not isinstance(dim, list):
                raise GraphParserError(
                    self,
                    node,
                    "Expected dim arg to be list type but got: " + str(type(dim)),
                )
            if len(dim) != 0 and len(dim) != 1:
                raise GraphParserError(
                    self,
                    node,
                    "Expected dim arg to be of length 0 or 1 but got length "
                    + str(len(dim)),
                )
            if len(dim) == 1:
                dim_elem: Argument = dim[0]
                if not isinstance(dim_elem, int):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected dim arg element to be int type but got: "
                        + str(type(dim_elem)),
                    )
                axes: list[int] = [dim_elem]

        keepdims: bool = False
        if len(node.args) == 3:
            keepdim: Argument = node.args[2]
            if not isinstance(keepdim, bool):
                raise GraphParserError(
                    self,
                    node,
                    "Expected keepdim arg to be bool type but got: "
                    + str(type(keepdim)),
                )
            keepdims: bool = keepdim

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_reduce_op(
            self._op_type,
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            axes,
            keepdims,
            debug_info,
        )


register_module("aten.max.default", MinMaxParser("max"))
register_module("aten.amax.default", MinMaxParser("max"))
register_module("aten.min.default", MinMaxParser("min"))
register_module("aten.amin.default", MinMaxParser("min"))


class AnyParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 1 or len(node.args) > 3:
            raise GraphParserError(
                self,
                node,
                "Expected between 1 and 3 arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        debug_info: DebugInfo = self.get_debug_info(node)
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)

        # Perform a cast if necessary
        if self_info.element_type().primitive_type != PrimitiveType.Bool:
            self_type: Type = self_info.sdfg_type()
            bool_scalar: Scalar = Scalar(PrimitiveType.Bool)
            if isinstance(self_type, Pointer):
                intermediate_type: Type = Pointer(bool_scalar)
            elif isinstance(self_type, Scalar):
                intermediate_type: Type = bool_scalar
            else:
                raise GraphParserError(
                    self,
                    node,
                    "Expected pointer or scalar input type but got: "
                    + self_type.print(),
                )
            intermediate_tensor: Tensor = Tensor(bool_scalar, self_info.shape())
            intermediate_info: TensorInfo = self.create_intermediate_tensor_info(
                node,
                builder,
                metadata,
                intermediate_type,
                intermediate_tensor,
                [self_info],
            )
            builder.add_cast_op(
                self_info.container(),
                self_info.sdfg_tensor_type(),
                intermediate_info.container(),
                intermediate_tensor,
                debug_info,
            )
            self_info: TensorInfo = intermediate_info

        axes: list[int] = [i for i in range(len(self_info.shape()))]
        if len(node.args) >= 2:
            dim: Argument = node.args[1]
            if dim is None:
                pass
            elif isinstance(dim, int):
                axes: list[int] = [dim]
            elif isinstance(dim, list):
                if len(dim) > 0:
                    axes: list[int] = []
                    for dim_elem in dim:
                        if not isinstance(dim_elem, int):
                            raise GraphParserError(
                                self,
                                node,
                                "Expected dim arg element to be int type but got: "
                                + str(type(dim_elem)),
                            )
                        axes.append(dim_elem)
            else:
                raise GraphParserError(
                    self,
                    node,
                    "Expected dim arg to be int or list type but got: "
                    + str(type(dim)),
                )

        keepdims: bool = False
        if len(node.args) == 3:
            keepdim: Argument = node.args[2]
            if not isinstance(keepdim, bool):
                raise GraphParserError(
                    self,
                    node,
                    "Expected keepdim arg to be bool type but got: "
                    + str(type(keepdim)),
                )
            keepdims: bool = keepdim

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        result_tensor: Tensor = result_info.sdfg_tensor_type()

        # Special case: UInt8 result type
        if result_tensor.element_type.primitive_type == PrimitiveType.UInt8:
            result_tensor: Tensor = Tensor(
                Scalar(PrimitiveType.Bool),
                result_tensor.shape,
                result_tensor.strides,
                result_tensor.offset,
            )

        builder.add_reduce_op(
            "max",
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_tensor,
            axes,
            keepdims,
            debug_info,
        )


register_module("aten.any.default", AnyParser())
register_module("aten.any.dim", AnyParser())
register_module("aten.any.dims", AnyParser())


class AxesReductionParser(GraphParserModule):
    """Reductions taking ``(self, dim=None, keepdim=False, *, dtype=None)``, such as mean and sum."""

    _op_type: str
    _has_dims: bool

    def __init__(self, op_type: str, has_dims: bool) -> None:
        self._op_type: str = op_type
        self._has_dims: bool = has_dims

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if self._has_dims:
            if len(node.args) != 2 and len(node.args) != 3:
                raise GraphParserError(
                    self,
                    node,
                    "Expected between 2 and 3 arguments but got " + str(len(node.args)),
                )
            arg_2: Argument = node.args[1]
            if len(node.args) == 3:
                arg_3: Argument = node.args[2]
                if not isinstance(arg_3, bool):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected bool type for second argument but got: "
                        + str(type(arg_3)),
                    )
                keepdim: bool = arg_3
            else:
                keepdim: bool = False
        else:
            if len(node.args) != 1:
                raise GraphParserError(
                    self,
                    node,
                    "Expected exactly one argument but got " + str(len(node.args)),
                )
            keepdim: bool = False

        debug_info: DebugInfo = self.get_debug_info(node)
        if "dtype" in node.kwargs:
            dtype_arg: Argument = node.kwargs["dtype"]
            if not isinstance(dtype_arg, torch.dtype):
                raise GraphParserError(
                    self,
                    node,
                    "Expected torch.dtype for dtype kwarg but got: "
                    + str(type(dtype_arg)),
                )
            self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
            base_type: Scalar = self.determine_sdfg_scalar_type(node, dtype_arg)
            cast_tensor: Tensor = Tensor(base_type, self_info.shape())
            cast_info: TensorInfo = self.create_intermediate_tensor_info(
                node, builder, metadata, Pointer(base_type), cast_tensor, [self_info]
            )
            builder.add_cast_op(
                self_info.container(),
                self_info.sdfg_tensor_type(),
                cast_info.container(),
                cast_tensor,
                debug_info,
            )
            self_info: TensorInfo = cast_info
        elif len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        else:
            self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)

        # None and the empty list both reduce every dimension.
        if self._has_dims and (arg_2 is None or arg_2 == []):
            axes: list[int] = [i for i in range(len(self_info.shape()))]
        elif self._has_dims:
            if not isinstance(arg_2, list):
                raise GraphParserError(
                    self,
                    node,
                    "Expected list type as second argument but got: "
                    + str(type(arg_2)),
                )
            axes: list[int] = []
            for elem in arg_2:
                if not isinstance(elem, int):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected int type as element of second argument but got: "
                        + str(type(elem)),
                    )
                axes.append(elem)
        else:
            axes: list[int] = [i for i in range(len(self_info.shape()))]

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)

        builder.add_reduce_op(
            self._op_type,
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            axes,
            keepdim,
            debug_info,
        )


register_module("aten.mean.default", AxesReductionParser("mean", False))
register_module("aten.mean.dim", AxesReductionParser("mean", True))
register_module("aten.sum.default", AxesReductionParser("sum", False))
register_module("aten.sum.dim_IntList", AxesReductionParser("sum", True))


class TopKParser(GraphParserModule):
    """``values, indices = topk(x, k, dim, largest, sorted)``: the indices are the last k of a stable argsort along
    ``dim``, largest first (the first k for ``largest=False``), and the values the elements of ``x`` they select.
    The results are always sorted, as ``sorted=False`` allows; among equal elements the later comes first, an order
    PyTorch leaves unspecified."""

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 2 or len(node.args) > 5:
            raise GraphParserError(
                self,
                node,
                f"Expected between 2 and 5 arguments but got {len(node.args)}",
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(self, node, f"Unsupported kwargs: {node.kwargs}")
        k: Argument = node.args[1]
        dim: Argument = node.args[2] if len(node.args) > 2 else -1
        largest: Argument = node.args[3] if len(node.args) > 3 else True
        if not isinstance(k, int) or not isinstance(dim, int):
            raise GraphParserError(self, node, "Expected k and dim to be int constants")
        if not isinstance(largest, bool):
            raise GraphParserError(self, node, "Expected largest to be a bool constant")

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        values_info, indices_info = self.get_result_tensor_infos(
            2, node, builder, metadata
        )
        shape: list[str] = self_info.shape()
        rank: int = len(shape)
        if rank == 0 or not -rank <= dim < rank:
            raise GraphParserError(
                self, node, f"dim {dim} is out of bounds for rank {rank}"
            )
        dim %= rank
        debug_info: DebugInfo = self.get_debug_info(node)

        index_scalar: Scalar = Scalar(PrimitiveType.Int64)
        order_tensor: Tensor = Tensor(index_scalar, shape)
        order_info: TensorInfo = self.create_intermediate_tensor_info(
            node, builder, metadata, Pointer(index_scalar), order_tensor, [self_info]
        )
        builder.add_reduce_op(
            "argsort",
            self_info.container(),
            self_info.sdfg_tensor_type(),
            order_info.container(),
            order_tensor,
            [dim],
            False,
            debug_info,
        )

        def int64(base: str) -> str:
            name: str = builder.find_new_name(base)
            builder.add_container(name, index_scalar, False)
            return name

        def address(info: TensorInfo, index: list[str]) -> str:
            terms = [f"({i})*({s})" for i, s in zip(index, info.strides())]
            return " + ".join([f"({info.offset()})"] + terms)

        # One sequential loop per dimension, k along dim: small loops the targets may unroll into scalar loads.
        counters: list[str] = []
        for d in range(rank):
            counter: str = int64("_topk_i")
            builder.begin_for(
                counter, "0", str(k) if d == dim else shape[d], "1", debug_info
            )
            counters.append(counter)
        position: str = (
            f"({shape[dim]}) - 1 - {counters[dim]}" if largest else counters[dim]
        )
        source: list[str] = list(counters)
        source[dim] = position
        selected: str = int64("_topk_index")

        def assign(
            src: AccessNode,
            src_subset: str,
            dst: AccessNode,
            dst_subset: str,
            block: Block,
        ) -> None:
            tasklet: Tasklet = builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            builder.add_memlet(
                block,
                src,
                "void",
                tasklet,
                "_in",
                subset=src_subset,
                debug_info=debug_info,
            )
            builder.add_memlet(
                block,
                tasklet,
                "_out",
                dst,
                "void",
                subset=dst_subset,
                debug_info=debug_info,
            )

        block: Block = builder.add_block(debug_info)
        assign(
            builder.add_access(block, order_info.container(), debug_info),
            address(order_info, source),
            builder.add_access(block, selected, debug_info),
            "",
            block,
        )
        block = builder.add_block(debug_info)
        assign(
            builder.add_access(block, selected, debug_info),
            "",
            builder.add_access(block, indices_info.container(), debug_info),
            address(indices_info, counters),
            block,
        )
        source[dim] = selected
        block = builder.add_block(debug_info)
        assign(
            builder.add_access(block, self_info.container(), debug_info),
            address(self_info, source),
            builder.add_access(block, values_info.container(), debug_info),
            address(values_info, counters),
            block,
        )
        for _ in range(rank):
            builder.end_for()


register_module("aten.topk.default", TopKParser())
