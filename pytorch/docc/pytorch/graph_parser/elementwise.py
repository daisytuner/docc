"""
GraphParser modules for parsing elementwise operations.
"""

import torch.fx
from torch.fx.node import Argument

from typing import Any

from docc.sdfg import (
    DebugInfo,
    StructuredSDFGBuilder,
    Type,
    Tensor,
    Scalar,
    CMathFunction,
)

from docc.pytorch.graph_parser.utils import (
    GraphParserError,
    GraphParserModule,
    ContainerInfos,
    register_module,
)


class UnaryTensorOpParser(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

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
        builder.add_elementwise_unary_op(
            self.op_type,
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.abs.default", UnaryTensorOpParser("abs"))


class UnaryCMathTensorOpParser(GraphParserModule):
    func: CMathFunction

    def __init__(self, func: CMathFunction) -> None:
        self.func: CMathFunction = func

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
        builder.add_elementwise_unary_cmath_op(
            self.func,
            self_container,
            self_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.acos.default", UnaryCMathTensorOpParser(CMathFunction.acos))
register_module("aten.acosh.default", UnaryCMathTensorOpParser(CMathFunction.acosh))
register_module("aten.asin.default", UnaryCMathTensorOpParser(CMathFunction.asin))
register_module("aten.asinh.default", UnaryCMathTensorOpParser(CMathFunction.asinh))
register_module("aten.atan.default", UnaryCMathTensorOpParser(CMathFunction.atan))
register_module("aten.atanh.default", UnaryCMathTensorOpParser(CMathFunction.atanh))


class ElementwiseTensorOpParser(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

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
        other: str | tuple[str, Scalar] = self.get_arg_sdfg_value(
            node, container_info, 1
        )
        if isinstance(other, str):
            other_container: str = other
            other_tensor: Tensor = self.get_tensor_type(
                node, container_info, other_container
            )
        else:
            other_container: str = other[0]
            other_tensor: Tensor = Tensor(
                self.align_constant_type(node, other, self_tensor.element_type),
                [],
            )
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_op(
            self.op_type,
            self_container,
            self_tensor,
            other_container,
            other_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.div.Tensor", ElementwiseTensorOpParser("div"))
register_module("aten.mul.Tensor", ElementwiseTensorOpParser("mul"))


class ElementwiseTensorOpParserWithAlpha(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

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
        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        debug_info: DebugInfo = self.get_debug_info(node)
        if len(node.kwargs) == 0:
            intermediate: str | tuple[str, Scalar] = self.get_arg_sdfg_value(
                node, container_info, 1
            )
            if isinstance(intermediate, str):
                intermediate_container: str = intermediate
                intermediate_tensor: Tensor = self.get_tensor_type(
                    node, container_info, intermediate_container
                )
            else:
                intermediate_container: str = intermediate[0]
                intermediate_tensor: Tensor = Tensor(
                    self.align_constant_type(
                        node, intermediate, self_tensor.element_type
                    ),
                    [],
                )
        elif len(node.kwargs) == 1:
            other: str | tuple[str, Scalar] = self.get_arg_sdfg_value(
                node, container_info, 1
            )
            if isinstance(other, str):
                other_container: str = other
                other_tensor: Tensor = self.get_tensor_type(
                    node, container_info, other_container
                )
                other_is_scalar: bool = False
            else:
                other_container: str = other[0]
                other_tensor: Tensor = Tensor(
                    self.align_constant_type(node, other, self_tensor.element_type), []
                )
                other_is_scalar: bool = True
            if not "alpha" in node.kwargs:
                raise GraphParserError(
                    self,
                    node,
                    "Only 'alpha' in kwargs is supported but got: " + str(node.kwargs),
                )
            alpha: str | tuple[str, Scalar] = self.convert_arg_to_sdfg_value(
                node, container_info, node.kwargs["alpha"]
            )
            if isinstance(alpha, str):
                alpha_container: str = alpha
                alpha_type: Scalar = self.get_scalar_type(
                    node, container_info, alpha_container
                )
            else:
                alpha_container: str = alpha[0]
                alpha_type: Scalar = self.align_constant_type(
                    node, alpha, other_tensor.element_type
                )
            if not isinstance(alpha_type, Scalar):
                raise GraphParserError(
                    self,
                    node,
                    "Input 'alpha' must be a scalar but got " + alpha_type.print(),
                )
            alpha_tensor: Tensor = Tensor(alpha_type, [])
            if other_is_scalar:
                intermediate_type: Type = self_tensor.element_type
            else:
                intermediate_type: Type = container_info[self_container].sdfg_type()
            intermediate_tensor: Tensor = Tensor(
                other_tensor.element_type, other_tensor.shape
            )
            intermediate_container: str = self.create_intermediate_container(
                node, builder, container_info, intermediate_type, intermediate_tensor
            )
            builder.add_elementwise_op(
                "mul",
                alpha_container,
                alpha_tensor,
                other_container,
                other_tensor,
                intermediate_container,
                intermediate_tensor,
                debug_info,
            )
        else:
            raise GraphParserError(
                self, node, "Unsupported number of kwargs: " + str(len(node.kwargs))
            )
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        builder.add_elementwise_op(
            self.op_type,
            self_container,
            self_tensor,
            intermediate_container,
            intermediate_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.add.Tensor", ElementwiseTensorOpParserWithAlpha("add"))


class ElementwiseCMathTensorOpParser(GraphParserModule):
    func: CMathFunction

    def __init__(self, func: CMathFunction) -> None:
        self.func: CMathFunction = func

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
        other_container: str = self.get_arg_container(node, container_info, 1)
        other_tensor: Tensor = self.get_tensor_type(
            node, container_info, other_container
        )
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_cmath_op(
            self.func,
            self_container,
            self_tensor,
            other_container,
            other_tensor,
            result_container,
            result_tensor,
            debug_info,
        )


register_module(
    "aten.atan2.default", ElementwiseCMathTensorOpParser(CMathFunction.atan2)
)
