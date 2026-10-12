"""PyTorch GraphModule Parser

This module contains the PyTorch GraphModule Parser.
"""

import numpy as np
import torch
import torch.export
import torch.fx
import torch.utils._pytree
import torch._functorch._aot_autograd.descriptors
from torch.fx.node import Argument

from typing import Any
from typing_extensions import Buffer

from docc.sdfg import (
    StructuredSDFGBuilder,
    StructuredSDFG,
    Type,
    Tensor,
    Scalar,
    Pointer,
    DebugInfo,
    Block,
    ConstantNode,
    AccessNode,
    Tasklet,
    TaskletCode,
)

from docc.pytorch.graph_parser.utils import (
    TensorName,
    ContainerMemory,
    TensorInfo,
    ContainerInfo,
    TensorMetadata,
    GraphParserError,
    GraphParserBase,
    get_node_target_name,
    dispatch_to_module,
)

import docc.pytorch.graph_parser.blas
import docc.pytorch.graph_parser.builtin
import docc.pytorch.graph_parser.convolution
import docc.pytorch.graph_parser.creation
import docc.pytorch.graph_parser.elementwise
import docc.pytorch.graph_parser.nonlinear_activation
import docc.pytorch.graph_parser.normalization
import docc.pytorch.graph_parser.padding
import docc.pytorch.graph_parser.pooling
import docc.pytorch.graph_parser.reduction
import docc.pytorch.graph_parser.reshaping
import docc.pytorch.graph_parser.sparse
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
        self._arguments: list[Any] = []

        self.metadata: TensorMetadata = TensorMetadata()
        self.result_copy: dict[TensorName, TensorName] = {}

    def flatten_arg(self, arg: Argument) -> list[torch.fx.Node]:
        """
        Flattens a nested tuple/list to a list of torch.fx.Node's.
        Example: ((arg_0, arg_1), arg_2) -> ["arg_0", "arg_1", "arg_2"]
        """
        result: list[torch.fx.Node] = []
        if isinstance(arg, (tuple, list)):
            for elem in arg:
                result: list[torch.fx.Node] = result + self.flatten_arg(elem)
        elif isinstance(arg, torch.fx.Node):
            result.append(arg)
        return result

    def get_tensors_from_arg(self, arg: Argument) -> list[TensorName]:
        """
        Returns a list of all tensor names form an argument. Recursively calls itself for lists and
        tuples.
        """
        return [node.name for node in self.flatten_arg(arg)]

    def parse(self) -> None:
        """
        Parses the GraphModule (exported program) to a structured SDFG. This is done in two steps.
        The first step is the pre-parsing step, in which all tensor information are collected and
        written to the tensor metadata. Except for input and output arguments those tensors do not
        have an underlying container yet.
        The second step is the parsing step, in which all operations are actually translated to SDFG
        operations. Containers are conservatively create along the way and also stored in the tensor
        metadata. At the end all allocated memory is freed.
        """
        nodes = self.ep.graph_module.graph.nodes

        # Collect all outputs for out args
        for node in nodes:
            if node.op == "placeholder":
                self.parse_placeholder(node)
            elif node.op == "call_function":
                if get_node_target_name(node.target) in (
                    "aten._assert_tensor_metadata.default",
                ):
                    continue  # Skip output-less ops

                sdfg_tensors: tuple[Tensor | None, ...] = self.get_node_sdfg_tensors(
                    node
                )
                deps: list[TensorName] = self.get_tensors_from_arg(node.args)
                if len(sdfg_tensors) == 1:
                    self.metadata.add_tensor(
                        node.name, TensorInfo(node.name, sdfg_tensors[0])
                    )
                    for dep in deps:
                        self.metadata.add_dependency(node.name, dep)
                else:
                    contained_names: list[TensorName] = []
                    for i in range(len(sdfg_tensors)):
                        tensor_name: TensorName = f"{node.name}_{i}"
                        contained_names.append(tensor_name)
                        self.metadata.add_tensor(
                            tensor_name, TensorInfo(tensor_name, sdfg_tensors[i])
                        )
                        for dep in deps:
                            self.metadata.add_dependency(tensor_name, dep)
                    self.metadata.add_tensor_tuple(node.name, contained_names)
            elif node.op == "output":
                output_nodes: list[torch.fx.Node] = self.flatten_arg(node.args)
                for output_node in output_nodes:
                    if not output_node.name in self.ep.graph_signature.user_outputs:
                        continue

                    sdfg_type: Type = self.get_node_sdfg_type(output_node)
                    out_name: str = output_node.name
                    # FX names like "sin" would shadow C math functions in generated code.
                    out_container: str = f"__daisy__{out_name}"

                    if self.metadata.has_container(
                        out_name
                    ) or self.metadata.has_container(out_container):
                        continue

                    if not self.metadata.tensor(out_name).is_tight() or isinstance(
                        sdfg_type, Scalar
                    ):
                        if not isinstance(sdfg_type, Pointer):
                            sdfg_type: Type = Pointer(sdfg_type)
                        new_out_name: str = f"{out_name}_out"
                        self.builder.add_container(
                            new_out_name, sdfg_type, is_argument=True
                        )
                        self.metadata.add_container(
                            new_out_name,
                            ContainerInfo(
                                new_out_name, sdfg_type, ContainerMemory.OUT_ARGUMENT
                            ),
                        )
                        out_shape: list[str] = self.metadata.tensor(out_name).shape()
                        if len(out_shape) == 0:
                            out_shape: list[str] = ["1"]
                        new_out_tensor: Tensor = Tensor(
                            self.metadata.tensor(out_name).element_type(),
                            out_shape,
                        )
                        self.metadata.add_tensor(
                            new_out_name,
                            TensorInfo(new_out_name, new_out_tensor, new_out_name),
                        )
                        self.metadata.add_dependency(new_out_name, out_name)
                        self.result_copy[out_name] = new_out_name
                    else:
                        self.builder.add_container(
                            out_container, sdfg_type, is_argument=True
                        )
                        self.metadata.add_container(
                            out_container,
                            ContainerInfo(
                                out_container, sdfg_type, ContainerMemory.OUT_ARGUMENT
                            ),
                        )
                        self.metadata.tensor(out_name).set_container(out_container)

        for node in nodes:
            if node.op == "placeholder":
                self.metadata.live(self.metadata.tensor(node.name).container())
            elif node.op == "call_function":
                pass
                dispatch_to_module(
                    node,
                    self.builder,
                    self.metadata,
                )
            elif node.op == "output":
                self.parse_output(node)
            else:
                raise GraphParserError(self, node, "Unknown op kind: " + node.op)

        for name in self.metadata.containers_memory_managed():
            if isinstance(self.metadata.container(name).sdfg_type(), Pointer):
                self.builder.add_free_block(name)
            self.metadata.dead(name)

    def parse_placeholder(self, node: torch.fx.Node) -> None:
        """
        Parses a PyTorch placeholder operation by creating a tensor information, a container
        information and, an actual SDFG container for it. Notice, that all arguments of an SDFG must
        have C-strides. This is also ensured here.
        """
        if not "desc" in node.meta:
            raise GraphParserError(self, node, "Missing desc metadata in placeholder")
        desc: Any = node.meta["desc"]
        if isinstance(desc, torch._functorch._aot_autograd.descriptors.PlainAOTInput):
            # Normal argument
            if desc.idx >= len(self.example_input):
                raise GraphParserError(
                    self,
                    node,
                    f"No example input for placeholder {desc.idx}",
                )
            sdfg_type: Type = self.determine_sdfg_type(
                node, self.example_input[desc.idx]
            )
            sdfg_tensor: Tensor | None = self.determine_sdfg_tensor_type(
                node, self.example_input[desc.idx]
            )
            self._arguments.append(self.example_input[desc.idx])

            self.builder.add_container(node.name, sdfg_type, is_argument=True)
            self.metadata.add_container(
                node.name,
                ContainerInfo(node.name, sdfg_type, ContainerMemory.IN_ARGUMENT),
            )

            # The call always provides a tensor with C-strides. If the tensor has non C-strides we
            # enforce them here.
            if sdfg_tensor is None:
                contiguous_tensor: Tensor | None = None
            else:
                contiguous_tensor: Tensor | None = Tensor(
                    sdfg_tensor.element_type, sdfg_tensor.shape
                )
            self.metadata.add_tensor(
                node.name, TensorInfo(node.name, contiguous_tensor, node.name)
            )
        elif isinstance(
            desc, torch._functorch._aot_autograd.descriptors.BufferAOTInput
        ):
            # Constant argument
            sdfg_type: Type = self.get_node_sdfg_type(node)
            sdfg_tensor: Tensor | None = self.get_node_sdfg_tensor(node)
            if sdfg_tensor is None:
                raise GraphParserError(self, node, "Could not get SDFG tensor type")

            self.builder.add_container(node.name, sdfg_type)
            self.metadata.add_container(
                node.name,
                ContainerInfo(node.name, sdfg_type, ContainerMemory.UNMANAGED),
            )
            self.metadata.add_tensor(
                node.name, TensorInfo(node.name, sdfg_tensor, node.name)
            )

            if desc.target not in self.ep.constants:
                raise GraphParserError(
                    self,
                    node,
                    "Encountered constant argument that is not present in the constants dictionary: "
                    + desc.target,
                )
            from torch.fx._symbolic_trace import _ConstantAttributeType

            constant: _ConstantAttributeType = self.ep.constants[desc.target]
            if not isinstance(constant, torch.Tensor):
                raise GraphParserError(
                    self,
                    node,
                    "Expected constant to be torch.Tensor type but got: "
                    + str(type(constant)),
                )

            debug_info: DebugInfo = self.get_debug_info(node)
            if constant.shape in (torch.Size([]), torch.Size([1])):
                # A single-value constant
                block: Block = self.builder.add_block(debug_info)
                const_access: ConstantNode = self.builder.add_constant(
                    block, str(constant.item()), sdfg_tensor.element_type, debug_info
                )
                output_access: AccessNode = self.builder.add_access(
                    block, node.name, debug_info
                )
                tasklet: Tasklet = self.builder.add_tasklet(
                    block, TaskletCode.assign, ["_in"], ["_out"], debug_info
                )
                self.builder.add_memlet(
                    block, const_access, "void", tasklet, "_in", debug_info=debug_info
                )
                self.builder.add_memlet(
                    block,
                    tasklet,
                    "_out",
                    output_access,
                    "void",
                    subset="" if isinstance(sdfg_type, Scalar) else "0",
                    debug_info=debug_info,
                )
            else:
                # Multi-value constant
                constant_arr: np.ndarray = constant.detach().cpu().contiguous().numpy()
                constant_raw: Buffer = constant_arr.reshape(-1).view(np.uint8)  # type: ignore

                self.builder.add_load_const_op(
                    node.name, sdfg_type, constant_raw, debug_info
                )
        else:
            raise GraphParserError(
                self, node, "Unsupported desc metadata type: " + str(desc)
            )

    def parse_output(self, node: torch.fx.Node) -> None:
        """
        Parses a PyTorch output operation by setting the SDFG metadata about output arguments and
        their shape information.
        """
        outputs: list[TensorName] = self.get_tensors_from_arg(node.args)
        non_user_outputs: list[int] = []
        for i in range(len(outputs)):
            output: TensorName = outputs[i]
            if not output in self.ep.graph_signature.user_outputs:
                non_user_outputs.append(i)
                continue

            if output in self.result_copy:
                new_output: TensorName = self.result_copy[output]
                output_info: TensorInfo = self.metadata.tensor(output)
                new_output_info: TensorInfo = self.metadata.tensor(new_output)
                self.copy_output_tensor(
                    node, self.builder, self.metadata, output_info, new_output_info
                )
                outputs[i] = new_output

        for non_user_output in non_user_outputs:
            del outputs[non_user_output]

        output_containers: list[str] = [
            self.metadata.tensor(output).container() for output in outputs
        ]
        self.builder.add_metadata("output_args", ",".join(output_containers))
        for output, output_container in zip(outputs, output_containers):
            self.builder.add_metadata(
                f"{output_container}_shape",
                self.metadata.tensor(output).shape_str(),
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

    def get_arguments(self) -> tuple[Any, ...]:
        """Return all the arguments"""
        return tuple(self._arguments)
