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
    Block,
    AccessNode,
    Tasklet,
    TaskletCode,
    Pointer,
)

from docc.pytorch.graph_parser.utils import (
    ContainerInfos,
    GraphParserError,
    GraphParserModule,
    register_module,
)


class MeanParser(GraphParserModule):
    _has_dims: bool

    def __init__(self, has_dims: bool) -> None:
        self._has_dims: bool = has_dims

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
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
            self_container: str = self.get_arg_container(node, container_info, 0)
            self_tensor: Tensor = self.get_tensor_type(
                node, container_info, self_container
            )
            base_type: Scalar = self.determine_sdfg_scalar_type(node, dtype_arg)
            cast_tensor: Tensor = Tensor(base_type, self_tensor.shape)
            cast_container: str = self.create_intermediate_container(
                node,
                builder,
                container_info,
                Pointer(base_type),
                cast_tensor,
            )
            builder.add_cast_op(
                self_container, self_tensor, cast_container, cast_tensor, debug_info
            )
            self_container: str = cast_container
            self_tensor: Tensor = cast_tensor
        elif len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        else:
            self_container: str = self.get_arg_container(node, container_info, 0)
            self_tensor: Tensor = self.get_tensor_type(
                node, container_info, self_container
            )
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
            axes: list[int] = [i for i in range(len(self_tensor.shape))]
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        result_type: Type = container_info[result_container].sdfg_type()
        if len(result_tensor.shape) == 0 and isinstance(result_type, Pointer):
            intermediate_container: str = self.create_intermediate_container(
                node, builder, container_info, result_tensor.element_type, result_tensor
            )
            builder.add_reduce_op(
                "mean",
                self_container,
                self_tensor,
                intermediate_container,
                result_tensor,
                axes,
                keepdim,
                debug_info,
            )
            block: Block = builder.add_block(debug_info)
            intermediate_access: AccessNode = builder.add_access(
                block, intermediate_container, debug_info
            )
            result_access: AccessNode = builder.add_access(
                block, result_container, debug_info
            )
            tasklet: Tasklet = builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            builder.add_memlet(
                block,
                intermediate_access,
                "void",
                tasklet,
                "_in",
                debug_info=debug_info,
            )
            builder.add_memlet(
                block,
                tasklet,
                "_out",
                result_access,
                "void",
                subset="0",
                debug_info=debug_info,
            )
        else:
            builder.add_reduce_op(
                "mean",
                self_container,
                self_tensor,
                result_container,
                result_tensor,
                axes,
                keepdim,
                debug_info,
            )


register_module("aten.mean.default", MeanParser(False))
register_module("aten.mean.dim", MeanParser(True))
