"""
GraphParser modules for parsing operations to create tensors.
"""

import torch.fx

from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    Tensor,
    DebugInfo,
    PrimitiveType,
    Block,
    AccessNode,
    Tasklet,
    TaskletCode,
)

from docc.pytorch.graph_parser.utils import (
    GraphParserModule,
    ContainerInfos,
    GraphParserError,
    register_module,
)


class FullParser(GraphParserModule):
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

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        fill_value: str | tuple[str, Scalar] = self.get_arg_sdfg_value(
            node, container_info, 1
        )
        if isinstance(fill_value, str):
            fill_value_container: str = fill_value
            fill_value_type: Scalar = self.get_scalar_type(
                node, container_info, fill_value_container
            )
        else:
            fill_value_container: str = fill_value[0]
            fill_value_type: Scalar = self.align_constant_type(
                node, fill_value, result_tensor.element_type
            )

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_fill_op(
            fill_value_container,
            fill_value_type,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.full.default", FullParser())
register_module("aten.full_like.default", FullParser())


class ArangeParser(GraphParserModule):
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
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

        if len(node.args) not in (2, 3):
            raise GraphParserError(
                self,
                node,
                "Expected between 2 and 3 arguments but got " + str(len(node.args)),
            )

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )

        start_val = self.get_arg_sdfg_value(node, container_info, 0)
        end_val = self.get_arg_sdfg_value(node, container_info, 1)
        if len(node.args) == 3:
            step_val = self.get_arg_sdfg_value(node, container_info, 2)
        else:
            step_val = ("1", Scalar(PrimitiveType.Int64))

        start_container = start_val if isinstance(start_val, str) else start_val[0]
        start_type = (
            self.get_scalar_type(node, container_info, start_container)
            if isinstance(start_val, str)
            else self.align_constant_type(node, start_val, result_tensor.element_type)
        )

        end_container = end_val if isinstance(end_val, str) else end_val[0]
        end_type = (
            self.get_scalar_type(node, container_info, end_container)
            if isinstance(end_val, str)
            else self.align_constant_type(node, end_val, result_tensor.element_type)
        )

        step_container = step_val if isinstance(step_val, str) else step_val[0]
        step_type = (
            self.get_scalar_type(node, container_info, step_container)
            if isinstance(step_val, str)
            else self.align_constant_type(node, step_val, result_tensor.element_type)
        )

        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_arange(
            start_container,
            start_type,
            end_container,
            end_type,
            step_container,
            step_type,
            result_container,
            result_tensor,
            debug_info,
        )


register_module("aten.arange.start_step", ArangeParser())


class ScalarTensorParser(GraphParserModule):
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

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)

        s: str | tuple[str, Scalar] = self.get_arg_sdfg_value(node, container_info, 0)
        block: Block = builder.add_block(debug_info)

        if isinstance(s, str):
            s_type: Scalar = self.get_scalar_type(node, container_info, s)
            s_access: AccessNode = builder.add_access(block, s, debug_info)
        else:
            s_type: Scalar = self.align_constant_type(
                node, s, result_tensor.element_type
            )
            s_access: AccessNode = builder.add_constant(block, s[0], s_type, debug_info)

        result_access: AccessNode = builder.add_access(
            block, result_container, debug_info
        )
        tasklet: Tasklet = builder.add_tasklet(
            block, TaskletCode.assign, ["_in"], ["_out"], debug_info
        )
        builder.add_memlet(
            block, s_access, "void", tasklet, "_in", type=s_type, debug_info=debug_info
        )
        builder.add_memlet(
            block,
            tasklet,
            "_out",
            result_access,
            "void",
            type=result_tensor,
            debug_info=debug_info,
        )


register_module("aten.scalar_tensor.default", ScalarTensorParser())
