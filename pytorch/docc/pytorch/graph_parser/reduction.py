"""
GraphParser modules for parsing reduction operations.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import (
    StructuredSDFGBuilder,
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


class MeanParser(GraphParserModule):
    _has_dims: bool

    def __init__(self, has_dims: bool) -> None:
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

        if self._has_dims:
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
            "mean",
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            axes,
            keepdim,
            debug_info,
        )


register_module("aten.mean.default", MeanParser(False))
register_module("aten.mean.dim", MeanParser(True))
