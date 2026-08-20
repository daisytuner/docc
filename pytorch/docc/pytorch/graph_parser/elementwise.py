"""
GraphParser modules for parsing elementwise operations.
"""

import torch.fx

from docc.sdfg import (
    DebugInfo,
    StructuredSDFGBuilder,
    Tensor,
    CMathFunction,
    TaskletCode,
)

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserError,
    GraphParserModule,
    register_module,
    primitive_type_is_floating_point,
    primitive_type_is_integer,
)


class UnaryTensorOpParser(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_unary_op(
            self.op_type,
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.abs.default", UnaryTensorOpParser("abs"))
register_module("aten.logical_not.default", UnaryTensorOpParser("logical_not"))
register_module("aten.sigmoid.default", UnaryTensorOpParser("sigmoid"))
register_module("aten.rsqrt.default", UnaryTensorOpParser("rsqrt"))


class NegParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        if primitive_type_is_floating_point(self_info.element_type().primitive_type):
            builder.add_elementwise_tasklet_op(
                TaskletCode.fp_neg,
                [self_info.container()],
                [self_info.sdfg_tensor_type()],
                result_info.container(),
                result_info.sdfg_tensor_type(),
                debug_info,
            )
        elif primitive_type_is_integer(self_info.element_type().primitive_type):
            builder.add_elementwise_op(
                "sub",
                "0",
                Tensor(self_info.element_type(), []),
                self_info.container(),
                self_info.sdfg_tensor_type(),
                result_info.container(),
                result_info.sdfg_tensor_type(),
                debug_info,
            )
        else:
            raise GraphParserError(
                self, node, "Neither floating point nor integer type"
            )


register_module("aten.neg.default", NegParser())


class UnaryCMathTensorOpParser(GraphParserModule):
    func: CMathFunction

    def __init__(self, func: CMathFunction) -> None:
        self.func: CMathFunction = func

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
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
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_unary_cmath_op(
            self.func,
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.acos.default", UnaryCMathTensorOpParser(CMathFunction.acos))
register_module("aten.acosh.default", UnaryCMathTensorOpParser(CMathFunction.acosh))
register_module("aten.asin.default", UnaryCMathTensorOpParser(CMathFunction.asin))
register_module("aten.asinh.default", UnaryCMathTensorOpParser(CMathFunction.asinh))
register_module("aten.atan.default", UnaryCMathTensorOpParser(CMathFunction.atan))
register_module("aten.atanh.default", UnaryCMathTensorOpParser(CMathFunction.atanh))
register_module("aten.cos.default", UnaryCMathTensorOpParser(CMathFunction.cos))
register_module("aten.sin.default", UnaryCMathTensorOpParser(CMathFunction.sin))


class ElementwiseTensorOpParser(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

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
        other_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 1, align_constant_type=self_info.element_type()
            )
        )

        self_tensor: Tensor = self_info.sdfg_tensor_type()
        other_tensor: Tensor = other_info_or_const.sdfg_tensor_type()
        if len(self_info.shape()) != len(other_info_or_const.shape()):
            self_tensor, other_tensor = self.align_elementwise_tensors(
                node, self_tensor, other_tensor
            )

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_op(
            self.op_type,
            self_info.container(),
            self_tensor,
            other_info_or_const.container(),
            other_tensor,
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.div.Tensor", ElementwiseTensorOpParser("div"))
register_module("aten.mul.Tensor", ElementwiseTensorOpParser("mul"))
register_module("aten.mul.Scalar", ElementwiseTensorOpParser("mul"))
register_module("aten.pow.Tensor_Scalar", ElementwiseTensorOpParser("pow"))
register_module("aten.pow.Tensor_Tensor", ElementwiseTensorOpParser("pow"))


class ElementwiseTaskletOpParser(GraphParserModule):
    fp_code: TaskletCode
    int_code: TaskletCode

    def __init__(self, fp_code: TaskletCode, int_code: TaskletCode) -> None:
        self.fp_code: TaskletCode = fp_code
        self.int_code: TaskletCode = int_code

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
        other_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 1, align_constant_type=self_info.element_type()
            )
        )
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)

        is_float = primitive_type_is_floating_point(
            self_info.element_type().primitive_type
        )
        is_integer = primitive_type_is_integer(self_info.element_type().primitive_type)
        if is_float:
            tasklet_code = self.fp_code
        elif is_integer:
            tasklet_code = self.int_code
        else:
            raise GraphParserError(
                self, node, "Unsupported primitive type for elementwise tasklet"
            )

        builder.add_elementwise_tasklet_op(
            tasklet_code,
            [self_info.container(), other_info_or_const.container()],
            [self_info.sdfg_tensor_type(), other_info_or_const.sdfg_tensor_type()],
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module(
    "aten.eq.Tensor", ElementwiseTaskletOpParser(TaskletCode.fp_oeq, TaskletCode.int_eq)
)
register_module(
    "aten.eq.Scalar", ElementwiseTaskletOpParser(TaskletCode.fp_oeq, TaskletCode.int_eq)
)
register_module(
    "aten.le.Tensor",
    ElementwiseTaskletOpParser(TaskletCode.fp_ole, TaskletCode.int_sle),
)
register_module(
    "aten.le.Scalar",
    ElementwiseTaskletOpParser(TaskletCode.fp_ole, TaskletCode.int_sle),
)
register_module(
    "aten.bitwise_and.Tensor",
    ElementwiseTaskletOpParser(TaskletCode.int_and, TaskletCode.int_and),
)


class ElementwiseTensorOpParserWithAlpha(GraphParserModule):
    op_type: str

    def __init__(self, op_type: str) -> None:
        self.op_type: str = op_type

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
        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        debug_info: DebugInfo = self.get_debug_info(node)
        if len(node.kwargs) == 0:
            intermediate_info_or_const: TensorInfo | TensorConstant = (
                self.get_arg_tensor_info_or_constant(
                    node, metadata, 1, align_constant_type=self_info.element_type()
                )
            )
        elif len(node.kwargs) == 1:
            other_info_or_const: TensorInfo | TensorConstant = (
                self.get_arg_tensor_info_or_constant(
                    node, metadata, 1, align_constant_type=self_info.element_type()
                )
            )
            if not "alpha" in node.kwargs:
                raise GraphParserError(
                    self,
                    node,
                    "Only 'alpha' in kwargs is supported but got: " + str(node.kwargs),
                )
            alpha_info_or_const: TensorInfo | TensorConstant = (
                self.convert_arg_to_tensor_info_or_constant(
                    node,
                    metadata,
                    node.kwargs["alpha"],
                    align_constant_type=other_info_or_const.element_type(),
                )
            )
            intermediate_tensor: Tensor = Tensor(
                other_info_or_const.element_type(), other_info_or_const.shape()
            )
            intermediate_info_or_const: TensorInfo | TensorConstant = (
                self.create_intermediate_tensor_info(
                    node,
                    builder,
                    metadata,
                    other_info_or_const.sdfg_type(),
                    intermediate_tensor,
                    [alpha_info_or_const, other_info_or_const],
                )
            )
            builder.add_elementwise_op(
                "mul",
                alpha_info_or_const.container(),
                alpha_info_or_const.sdfg_tensor_type(),
                other_info_or_const.container(),
                other_info_or_const.sdfg_tensor_type(),
                intermediate_info_or_const.container(),
                intermediate_tensor,
                debug_info,
            )
        else:
            raise GraphParserError(
                self, node, "Unsupported number of kwargs: " + str(len(node.kwargs))
            )

        self_tensor: Tensor = self_info.sdfg_tensor_type()
        intermediate_tensor: Tensor = intermediate_info_or_const.sdfg_tensor_type()
        if len(self_info.shape()) != len(intermediate_info_or_const.shape()):
            self_tensor, intermediate_tensor = self.align_elementwise_tensors(
                node, self_tensor, intermediate_tensor
            )

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        builder.add_elementwise_op(
            self.op_type,
            self_info.container(),
            self_tensor,
            intermediate_info_or_const.container(),
            intermediate_tensor,
            result_info.container(),
            result_info.sdfg_tensor_type(),
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
        other_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 1)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_elementwise_cmath_op(
            self.func,
            self_info.container(),
            self_info.sdfg_tensor_type(),
            other_info.container(),
            other_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module(
    "aten.atan2.default", ElementwiseCMathTensorOpParser(CMathFunction.atan2)
)
