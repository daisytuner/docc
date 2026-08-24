"""
GraphParser modules for parsing operations to create tensors.
"""

import torch.fx

from docc.sdfg import (
    StructuredSDFGBuilder,
    DebugInfo,
    Block,
    AccessNode,
    Tasklet,
    TaskletCode,
)

from docc.pytorch.graph_parser.utils import (
    TensorInfo,
    TensorConstant,
    TensorMetadata,
    GraphParserModule,
    GraphParserError,
    register_module,
)


class FullParser(GraphParserModule):
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
        if not set(node.kwargs.keys()).issubset(
            {"dtype", "layout", "device", "pin_memory", "memory_format"}
        ):
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self.get_kwarg_dtype(node)
        self.get_kwarg_layout(node)
        self.get_kwarg_device(node)
        self.get_kwarg_pin_memory(node)
        self.get_kwarg_memory_format(node)

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        fill_value_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 1, align_constant_type=result_info.element_type()
            )
        )

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_fill_op(
            fill_value_info_or_const.container(),
            fill_value_info_or_const.element_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.full.default", FullParser())
register_module("aten.full_like.default", FullParser())


class ArangeParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        if len(node.args) < 2 or len(node.args) > 3:
            raise GraphParserError(
                self,
                node,
                "Expected between 2 and 3 arguments but got " + str(len(node.args)),
            )

        if not set(node.kwargs.keys()).issubset(
            {"dtype", "layout", "device", "pin_memory"}
        ):
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self.get_kwarg_dtype(node)
        self.get_kwarg_layout(node)
        self.get_kwarg_device(node)
        self.get_kwarg_pin_memory(node)

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        start_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 0, align_constant_type=result_info.element_type()
            )
        )
        end_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 1, align_constant_type=result_info.element_type()
            )
        )
        if len(node.args) == 3:
            step_info_or_const: TensorInfo | TensorConstant = (
                self.get_arg_tensor_info_or_constant(
                    node, metadata, 2, align_constant_type=result_info.element_type()
                )
            )
        else:
            step_info_or_const: TensorInfo | TensorConstant = TensorConstant(
                "1", result_info.element_type()
            )

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_arange(
            start_info_or_const.container(),
            start_info_or_const.element_type(),
            end_info_or_const.container(),
            end_info_or_const.element_type(),
            step_info_or_const.container(),
            step_info_or_const.element_type(),
            result_info.container(),
            result_info.sdfg_tensor_type(),
            debug_info,
        )


register_module("aten.arange.start_step", ArangeParser())


class ScalarTensorParser(GraphParserModule):
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
                f"Expected exactly one argument but got {len(node.args)}",
            )
        if not set(node.kwargs.keys()).issubset(
            {"dtype", "layout", "device", "pin_memory"}
        ):
            raise GraphParserError(self, node, f"Unsupported kwargs: {node.kwargs}")

        self.get_kwarg_dtype(node)
        self.get_kwarg_layout(node)
        self.get_kwarg_device(node)
        self.get_kwarg_pin_memory(node)

        result_info: TensorInfo = self.get_result_tensor_info(node, builder, metadata)
        s_info_or_const: TensorInfo | TensorConstant = (
            self.get_arg_tensor_info_or_constant(
                node, metadata, 0, align_constant_type=result_info.element_type()
            )
        )

        debug_info: DebugInfo = self.get_debug_info(node)
        block: Block = builder.add_block(debug_info)

        if isinstance(s_info_or_const, TensorConstant):
            s_access: AccessNode = builder.add_constant(
                block,
                s_info_or_const.value(),
                s_info_or_const.sdfg_scalar(),
                debug_info,
            )
        else:
            s_access: AccessNode = builder.add_access(
                block, s_info_or_const.container(), debug_info
            )

        result_access: AccessNode = builder.add_access(
            block, result_info.container(), debug_info
        )
        tasklet: Tasklet = builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        builder.add_memlet(
            block,
            s_access,
            "void",
            tasklet,
            "_in",
            type=s_info_or_const.sdfg_type(),
            debug_info=debug_info,
        )
        builder.add_memlet(
            block,
            tasklet,
            "_out",
            result_access,
            "void",
            type=result_info.sdfg_tensor_type(),
            debug_info=debug_info,
        )


register_module("aten.scalar_tensor.default", ScalarTensorParser())
