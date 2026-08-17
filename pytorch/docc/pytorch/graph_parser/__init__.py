"""PyTorch GraphModule Parser

This module contains the PyTorch GraphModule Parser.
"""

import torch
import torch.export
import torch.fx
import torch.utils._pytree
from torch.fx.node import Argument

from typing import Any

from docc.sdfg import (
    StructuredSDFGBuilder,
    StructuredSDFG,
    Type,
    Tensor,
    DebugInfo,
)

from docc.pytorch.graph_parser.utils import (
    ContainerInfoBase,
    ContainerInfo,
    ContainerPreInfo,
    ContainerRefInfo,
    ContainerInfos,
    GraphParserError,
    GraphParserBase,
    dispatch_to_pre_module,
    dispatch_to_module,
)
import docc.pytorch.graph_parser.blas
import docc.pytorch.graph_parser.builtin
import docc.pytorch.graph_parser.convolution
import docc.pytorch.graph_parser.creation
import docc.pytorch.graph_parser.elementwise
import docc.pytorch.graph_parser.nonlinear_activation
import docc.pytorch.graph_parser.normalization
import docc.pytorch.graph_parser.pooling
import docc.pytorch.graph_parser.reduction
import docc.pytorch.graph_parser.reshaping
import docc.pytorch.graph_parser.tensor
import docc.pytorch.graph_parser.vision


class GraphParser(GraphParserBase):
    """
    This is the main PyTorch GraphModule Parser class. It creates a structured SDFG from a
    GraphModule obtained after calling torch.export and its example inputs. It dispatches to the
    GraphParser modules for parsing individual operations.
    """

    def __init__(
        self,
        name: str,
        ep: torch.export.ExportedProgram,
        example_input: tuple[Any, ...],
    ):
        """
        Intialization with the GraphModule after calling torch.export and the example inputs.
        """
        super().__init__()

        self.name: str = name
        self.ep: torch.export.ExportedProgram = ep
        self.example_input: tuple[Any, ...] = example_input

        self.builder: StructuredSDFGBuilder = StructuredSDFGBuilder("__docc_" + name)
        self._placeholder_index: int = 0

        self.container_info: ContainerInfos = ContainerInfos()

    def get_output_containers(
        self, node: torch.fx.Node, args: Argument, resolve: bool = True
    ) -> list[str]:
        """
        Flattens a nested tuple to a list and converts each PyTorch Argument to an SDFG container.
        Example: ((arg_0, arg_1), arg_2) -> ["arg_0", "arg_1", "arg_2"]
        """
        result = []
        if isinstance(args, tuple):
            for elem in list(args):
                result += self.get_output_containers(node, elem, resolve=resolve)
        else:
            result.append(
                self.convert_arg_to_container(
                    node, self.container_info, args, resolve=resolve
                )
            )
        return result

    def parse(self) -> None:
        """
        Parses the GraphModule (exported program) to a structured SDFG. This is done in two steps.
        The first step is the pre-parsing step, in which the container information are filled with
        data about "virtual" containers, i.e., containers that reference other containers in the
        same way that two PyTorch tensors can share the same underlying data. The second step is the
        parsing step, in which all operations are actually translated to SDFG operations. At the end
        all allocated memory is freed.
        """
        nodes = self.ep.graph_module.graph.nodes

        # Collect all outputs for out args
        for node in nodes:
            if node.op == "placeholder":
                self.container_info[node.name] = ContainerPreInfo(
                    node.name, in_argument=True
                )
            elif node.op == "call_function":
                dispatch_to_pre_module(node, self.builder, self.container_info)
            elif node.op == "output":
                output_containers: list[str] = self.get_output_containers(
                    node, node.args, resolve=False
                )
                for output_container in output_containers:
                    if output_container in self.container_info:
                        info: ContainerInfoBase = self.container_info[output_container]
                        if not isinstance(info, ContainerPreInfo):
                            raise GraphParserError(
                                self,
                                node,
                                "Expected ContainterPreInfo but got: "
                                + str(type(info)),
                            )
                        self.container_info[output_container] = ContainerPreInfo.copy(
                            info, out_argument=True
                        )
                    else:
                        self.container_info[output_container] = ContainerPreInfo(
                            output_container, out_argument=True
                        )

        # Parse all operations
        for node in nodes:
            if node.op == "placeholder":
                self.parse_placeholder(node)
            elif node.op == "call_function":
                dispatch_to_module(node, self.builder, self.container_info)
            elif node.op == "output":
                self.parse_output(node)
            else:
                raise GraphParserError(self, node, "Unknown op kind: " + node.op)

        for container in self.container_info.memory_managed():
            self.builder.add_free_block(container)

    def parse_placeholder(self, node: torch.fx.Node) -> None:
        """
        Parses a PyTorch placeholder operation by creating an SDFG container for it. Notice, that
        all arguments of an SDFG must have C-strides. This is also ensured here.
        """
        if self._placeholder_index >= len(self.example_input):
            raise GraphParserError(
                self,
                node,
                f"No example input for placeholder {self._placeholder_index}",
            )
        sdfg_types: tuple[Type, Tensor | None] = self.determine_sdfg_type(
            node, self.example_input[self._placeholder_index]
        )
        # The call always provides a tensor with C-strides. If the tensor has non C-strides we
        # enforce them here.
        if sdfg_types[1] is None:
            contiguous_tensor: Tensor | None = None
        else:
            contiguous_tensor: Tensor | None = Tensor(
                sdfg_types[1].element_type, sdfg_types[1].shape
            )
        self.builder.add_container(node.name, sdfg_types[0], True)
        self.container_info[node.name] = ContainerInfo(
            node.name, sdfg_types[0], contiguous_tensor, in_argument=True
        )
        self._placeholder_index += 1

    def parse_output(self, node: torch.fx.Node) -> None:
        """
        Parses a PyTorch output operation by setting the SDFG metadata about output arguments and
        their shape information.
        """
        raw_output_containers: list[str] = self.get_output_containers(
            node, node.args, resolve=False
        )
        output_containers: list[str] = []
        for i, container in enumerate(raw_output_containers):
            info: ContainerInfoBase = self.container_info[container]
            src_tensor: Tensor | None = info.sdfg_tensor_type()
            needs_copy: bool = False
            if not info.out_argument():
                needs_copy = True
            elif src_tensor is not None and not src_tensor.is_contiguous():
                needs_copy = True
            elif isinstance(info, ContainerRefInfo):
                needs_copy = True

            output_container: str = container + "_out"
            if output_container in self.container_info:
                self.copy_scalar_tensor(
                    node,
                    self.builder,
                    self.container_info,
                    container,
                    output_container,
                )
                output_containers.append(output_container)
            elif needs_copy and src_tensor is not None:
                base_container: str = (
                    info.ref().name()
                    if isinstance(info, ContainerRefInfo)
                    else info.name()
                )
                dst_tensor: Tensor = Tensor(src_tensor.element_type, src_tensor.shape)
                out_type: Type = info.sdfg_type()
                new_out_container: str = self.builder.find_new_name(
                    container + "_out"
                )
                self.builder.add_container(
                    new_out_container, out_type, is_argument=True
                )
                self.container_info[new_out_container] = ContainerInfo(
                    new_out_container, out_type, dst_tensor, out_argument=True
                )
                debug_info: DebugInfo = self.get_debug_info(node)
                self.builder.add_copy_op(
                    base_container,
                    src_tensor,
                    new_out_container,
                    dst_tensor,
                    debug_info,
                )
                output_containers.append(new_out_container)
            else:
                output_containers.append(container)

        self.builder.add_metadata("output_args", ",".join(output_containers))
        for output_container in output_containers:
            self.builder.add_metadata(
                f"{output_container}_shape",
                self.container_info.get_shape_str(output_container),
            )

        # Preserve the original PyTorch output structure (nested tuples, dicts,
        # dataclasses, single tensor, None leaves, ...) as a serialized pytree
        # spec. At runtime the flat outputs produced by the compiled artifact are
        # reassembled with this spec so the returned structure matches eager
        # PyTorch exactly.
        out_spec: torch.utils._pytree.TreeSpec | None = self.ep.call_spec.out_spec
        if out_spec is not None:
            self.builder.add_metadata(
                "output_pytree_spec", torch.utils._pytree.treespec_dumps(out_spec)
            )
            self.builder.add_metadata("output_num_leaves", str(out_spec.num_leaves))

    def to_sdfg(self) -> StructuredSDFG:
        """
        Detaches the structured SDFG from the internal structured SDFG builder and returns it.
        """
        return self.builder.move()
