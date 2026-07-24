"""
GraphParser modules for parsing operations to create tensors.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Scalar, Tensor, DebugInfo

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

        dtype: torch.dtype | None = None
        if "dtype" in node.kwargs:
            dtype_arg: Argument = node.kwargs["dtype"]
            if not dtype_arg is None:
                if not isinstance(dtype_arg, torch.dtype):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected dtype kwarg to be torch.dtype type but got: "
                        + str(type(dtype_arg)),
                    )
                dtype: torch.dtype | None = dtype_arg

        if "layout" in node.kwargs:
            layout_arg: Argument = node.kwargs["layout"]
            if not layout_arg is None:
                if isinstance(layout_arg, torch.layout):
                    if layout_arg != torch.strided:
                        raise GraphParserError(
                            self,
                            node,
                            "Only layout torch.strided is supported but got: "
                            + str(layout_arg),
                        )
                else:
                    raise GraphParserError(
                        self,
                        node,
                        "Expected layout kwarg to be torch.layout type but got: "
                        + str(type(layout_arg)),
                    )

        if "device" in node.kwargs:
            device_arg: Argument = node.kwargs["device"]
            if not device_arg is None:
                if isinstance(device_arg, torch.device):
                    if device_arg.type != "cpu":
                        raise GraphParserError(
                            self, node, "Currently only CPU device kwarg is supported"
                        )
                else:
                    raise GraphParserError(
                        self,
                        node,
                        "Expected device kwarg to be torch.device type but got: "
                        + str(type(device_arg)),
                    )

        if "pin_memory" in node.kwargs:
            pin_memory_arg: Argument = node.kwargs["pin_memory"]
            if not pin_memory_arg is None:
                if isinstance(pin_memory_arg, bool):
                    if pin_memory_arg:
                        raise GraphParserError(
                            self, node, "Currently pin_memory is unsupported"
                        )
                else:
                    raise GraphParserError(
                        self,
                        node,
                        "Expected pin_memory kwarg to be bool type but got: "
                        + str(type(pin_memory_arg)),
                    )

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
