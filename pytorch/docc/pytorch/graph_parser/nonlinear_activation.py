"""
GraphParser modules for parsing non-linear activation functions.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, DebugInfo, PrimitiveType, Scalar

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


class ReLUParser(GraphParserModule):
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
        builder.add_relu(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.relu.default", ReLUParser())


class GELUParser(GraphParserModule):
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

        tanh_approx: bool = False
        if "approximate" in node.kwargs:
            approximate: Argument = node.kwargs["approximate"]
            if not isinstance(approximate, str):
                raise GraphParserError(
                    self,
                    node,
                    "Expected approximate kwarg to be str type but got: "
                    + str(type(approximate)),
                )
            if not approximate in ["none", "tanh"]:
                raise GraphParserError(
                    self, node, "Unknown approximation: " + approximate
                )
            if approximate == "tanh":
                tanh_approx: bool = True
        elif len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_gelu(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            tanh_approx,
            debug_info,
        )


register_module("aten.gelu.default", GELUParser())


class ClampParser(GraphParserModule):
    BOUNDS: tuple[str, ...] = ("min", "max")

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 1 or len(node.args) > 1 + len(self.BOUNDS):
            raise GraphParserError(
                self,
                node,
                "Expected one to three arguments but got " + str(len(node.args)),
            )
        if len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        self_prim = self_info.element_type().primitive_type
        if self_prim == PrimitiveType.Bool or not (
            primitive_type_is_floating_point(self_prim)
            or primitive_type_is_integer(self_prim)
        ):
            raise GraphParserError(
                self,
                node,
                "Expected a floating point or integer input but got " + str(self_prim),
            )
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        result_prim = result_info.element_type().primitive_type
        if self_prim != result_prim:
            raise GraphParserError(
                self,
                node,
                f"Expected matching input and result types but got {self_prim} and {result_prim}",
            )

        bounds: list[tuple[str, Scalar | None]] = []
        for i, name in enumerate(self.BOUNDS):
            value: Argument = node.args[i + 1] if i + 1 < len(node.args) else None
            if value is None:
                bounds.append(("", None))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GraphParserError(
                    self,
                    node,
                    f"Expected {name} to be int or float type but got: {type(value)}",
                )
            constant: TensorConstant = self.convert_arg_to_tensor_constant(node, value)
            constant_type: Scalar = self.align_constant_type(
                node, constant, self_info.element_type()
            )
            bounds.append((constant.value(), constant_type))
        if all(bound_type is None for _, bound_type in bounds):
            raise GraphParserError(self, node, "Expected at least one of min and max")

        (min_value, min_type), (max_value, max_type) = bounds
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_clamp(
            self_info.container(),
            self_info.sdfg_tensor_type(),
            min_value,
            min_type,
            max_value,
            max_type,
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.clamp.default", ClampParser())


class SoftmaxParser(GraphParserModule):
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

        dim: Argument = node.args[1]
        if not isinstance(dim, int):
            raise GraphParserError(
                self, node, "Expected dim arg to be int type but got: " + str(type(dim))
            )
        half_to_float: Argument = node.args[2]
        if not isinstance(half_to_float, bool):
            raise GraphParserError(
                self,
                node,
                "Expected half_to_float arg to be bool type but got: "
                + str(type(half_to_float)),
            )
        if half_to_float:
            raise GraphParserError(
                self, node, "Currently setting half_to_float arg is unsupported"
            )

        self_info: TensorInfo = self.get_arg_tensor_info(node, metadata, 0)
        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_reduce_op(
            "softmax",
            self_info.container(),
            self_info.sdfg_tensor_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            [dim],
            False,
            debug_info,
        )


register_module("aten._softmax.default", SoftmaxParser())
