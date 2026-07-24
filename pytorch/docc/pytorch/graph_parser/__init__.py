"""PyTorch GraphModule Parser

This module contains the PyTorch GraphModule Parser.
"""

import torch
import torch.export
import torch.fx
from torch.fx.node import Argument

from typing import Any

from docc.sdfg import StructuredSDFGBuilder, StructuredSDFG, Type, Tensor

from docc.pytorch.graph_parser.utils import (
    ContainerInfoBase,
    ContainerInfo,
    ContainerPreInfo,
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

    def get_output_containers(self, node: torch.fx.Node, args: Argument) -> list[str]:
        """
        Flattens a nested tuple to a list and converts each PyTorch Argument to an SDFG container.
        Example: ((arg_0, arg_1), arg_2) -> ["arg_0", "arg_1", "arg_2"]
        """
        result = []
        if isinstance(args, tuple):
            for elem in list(args):
                result += self.get_output_containers(node, elem)
        else:
            result.append(
                self.convert_arg_to_container(node, self.container_info, args)
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
                    node, node.args
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

        for node in nodes:
            if node.op == "placeholder":
                self.parse_placeholder(node)
            elif node.op == "call_function":
                dispatch_to_module(
                    node,
                    self.builder,
                    self.container_info,
                )
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
        output_containers: list[str] = self.get_output_containers(node, node.args)
        self.builder.add_metadata("output_args", ",".join(output_containers))
        for output_container in output_containers:
            if not self.container_info[output_container].out_argument():
                raise GraphParserError(
                    self, node, "Unsupported for now; needs explicit copy"
                )
            self.builder.add_metadata(
                f"{output_container}_shape",
                self.container_info.get_shape_str(output_container),
            )

    def to_sdfg(self) -> StructuredSDFG:
        """
        Detaches the structured SDFG from the internal structured SDFG builder and returns it.
        """
        return self.builder.move()
