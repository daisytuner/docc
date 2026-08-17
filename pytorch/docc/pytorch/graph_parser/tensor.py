"""
GraphParser module for parsing operations performed directly on a tensor object.
"""

import torch.fx
from torch.fx.node import Argument

from docc.sdfg import StructuredSDFGBuilder, Tensor, Scalar, DebugInfo

from docc.pytorch.graph_parser.utils import (
    ContainerInfoBase,
    ContainerPreInfo,
    ContainerInfos,
    GraphParserError,
    GraphParserModule,
    register_pre_module,
    register_module,
)


class AssertTensorMetadataParser(GraphParserModule):
    """
    This does not change anything. It is basically a PyTorch assertion. We also verify that
    everything is set correctly.
    """

    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        if len(node.args) != 4:
            raise GraphParserError(
                self,
                node,
                "Expected exactly 4 arguments but got: " + str(len(node.args)),
            )
        if not set(node.kwargs.keys()).issubset({"device", "layout"}):
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)

        size: Argument = node.args[1]
        if not size is None:
            if not isinstance(size, list):
                raise GraphParserError(
                    self,
                    node,
                    "Expected size arg to be list type but got: " + str(type(size)),
                )
            dims: int = len(size)
            if len(self_tensor.shape) != dims:
                raise GraphParserError(
                    self,
                    node,
                    f"Mismatched size! Expected {size} but got {self_tensor.shape}",
                )
            for i in range(dims):
                elem: Argument = size[i]
                if not isinstance(elem, int):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected size arg element to be int type but got: "
                        + str(type(elem)),
                    )
                if str(elem) != self_tensor.shape[i]:
                    raise GraphParserError(
                        self,
                        node,
                        f"Mismatched size! Expected {size} but got {self_tensor.shape}",
                    )

        stride: Argument = node.args[2]
        if not stride is None:
            if not isinstance(stride, list):
                raise GraphParserError(
                    self,
                    node,
                    "Expected stride arg to be list type but got: " + str(type(stride)),
                )
            dims: int = len(stride)
            if len(self_tensor.strides) != dims:
                raise GraphParserError(
                    self,
                    node,
                    f"Mismatched stride! Expected {stride} but got {self_tensor.strides}",
                )
            for i in range(dims):
                elem: Argument = stride[i]
                if not isinstance(elem, int):
                    raise GraphParserError(
                        self,
                        node,
                        "Expected stride arg element to be int type but got: "
                        + str(type(elem)),
                    )
                if str(elem) != self_tensor.strides[i]:
                    raise GraphParserError(
                        self,
                        node,
                        f"Mismatched stride! Expected {stride} but got {self_tensor.strides}",
                    )

        dtype: Argument = node.args[3]
        if not dtype is None:
            if not isinstance(dtype, torch.dtype):
                raise GraphParserError(
                    self,
                    node,
                    "Expected dtype arg to be torch.dtype type but got: "
                    + str(type(dtype)),
                )
            scalar: Scalar = self.determine_sdfg_scalar_type(node, dtype)
            if self_tensor.element_type.primitive_type != scalar.primitive_type:
                raise GraphParserError(
                    self,
                    node,
                    f"Mismatched dtype! Expected {scalar} but got {self_tensor.element_type}",
                )

        self.get_kwarg_device(node)
        self.get_kwarg_layout(node)


register_module("aten._assert_tensor_metadata.default", AssertTensorMetadataParser())


class CloneParser(GraphParserModule):
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
                "Expected exactly 2 arguments but got: " + str(len(node.args)),
            )
        if not set(node.kwargs.keys()).issubset({"memory_format"}):
            raise GraphParserError(self, node, f"Unsupported kwargs: {node.kwargs}")

        self.get_kwarg_memory_format(node)

        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)
        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )
        debug_info: DebugInfo = self.get_debug_info(node)
        builder.add_copy_op(
            self_container, self_tensor, result_container, result_tensor, debug_info
        )


register_module("aten.clone.default", CloneParser())


class ViewParser(GraphParserModule):
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


register_pre_module("aten.view.default", ViewParser())
register_module("aten.view.default", ViewParser())


class ExpandParser(GraphParserModule):
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
        if "implicit" in node.kwargs:
            pass  # Nothing to do
        elif len(node.kwargs) != 0:
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
        if "implicit" in node.kwargs:
            pass  # Nothing to do
        elif len(node.kwargs) != 0:
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )
        self.update_container_types(node, builder, container_info, node.name)


register_pre_module("aten.expand.default", ExpandParser())
register_module("aten.expand.default", ExpandParser())


class ToCopyParser(GraphParserModule):
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
        if not set(node.kwargs.keys()).issubset(
            {"dtype", "layout", "device", "pin_memory", "non_blocking", "memory_format"}
        ):
            raise GraphParserError(
                self, node, "Unsupported kwargs: " + str(node.kwargs)
            )

        self_container: str = self.get_arg_container(node, container_info, 0)
        self_tensor: Tensor = self.get_tensor_type(node, container_info, self_container)

        self.get_kwarg_layout(node)
        self.get_kwarg_device(node)
        self.get_kwarg_pin_memory(node)
        self.get_kwarg_memory_format(node)
        if "non_blocking" in node.kwargs:
            non_blocking: Argument = node.kwargs["non_blocking"]
            if not isinstance(non_blocking, bool):
                raise GraphParserError(
                    self,
                    node,
                    "Expected non_blocking kwarg to be bool type but got: "
                    + str(type(non_blocking)),
                )
            if non_blocking:
                raise GraphParserError(
                    self, node, "Currently non_blocking is unsupported"
                )

        result_container: str = self.get_result_container(node, builder, container_info)
        result_tensor: Tensor = self.get_tensor_type(
            node, container_info, result_container
        )

        cast: bool = False
        dtype: torch.dtype | None = self.get_kwarg_dtype(node)
        if not dtype is None:
            dtype_scalar: Scalar = self.determine_sdfg_scalar_type(node, dtype)
            if dtype_scalar.primitive_type != result_tensor.primitive_type:
                raise GraphParserError(
                    self,
                    node,
                    f"dtype mismatch! Expected {dtype_scalar} but got {result_tensor.element_type}",
                )
            cast: bool = True

        debug_info: DebugInfo = self.get_debug_info(node)
        if cast:
            builder.add_cast_op(
                self_container, self_tensor, result_container, result_tensor, debug_info
            )
        else:
            builder.add_copy_op(
                self_container, self_tensor, result_container, result_tensor, debug_info
            )


register_module("aten._to_copy.default", ToCopyParser())
