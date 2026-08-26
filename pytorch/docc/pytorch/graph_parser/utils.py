"""PyTorch GraphModule Parser utilities

This module contains the utilities and base classes for the PyTorch GraphModule Parser.
"""

import torch
import torch.fx
import torch.fx.passes.shape_prop
from torch.fx.node import Argument, Target

from typing import Any
from abc import ABC, abstractmethod
import math
from enum import Enum

from docc.sdfg import (
    Type,
    PrimitiveType,
    Scalar,
    Pointer,
    Tensor,
    DebugInfo,
    StructuredSDFGBuilder,
    Block,
    AccessNode,
    Tasklet,
    TaskletCode,
)


def primitive_type_is_signed_integer(primitive_type: PrimitiveType) -> bool:
    """Returns True iff the given SDFG primitive type is a signed integer"""
    return primitive_type in [
        PrimitiveType.Int8,
        PrimitiveType.Int16,
        PrimitiveType.Int32,
        PrimitiveType.Int64,
        PrimitiveType.Int128,
    ]


def primitive_type_is_unsigned_integer(primitive_type: PrimitiveType) -> bool:
    """Returns True iff the given SDFG primitive type is an unsigned integer"""
    return primitive_type in [
        PrimitiveType.Bool,
        PrimitiveType.UInt8,
        PrimitiveType.UInt16,
        PrimitiveType.UInt32,
        PrimitiveType.UInt64,
        PrimitiveType.UInt128,
    ]


def primitive_type_is_integer(primitive_type: PrimitiveType) -> bool:
    """Returns True iff the given SDFG primitive type is an integer"""
    return primitive_type_is_signed_integer(
        primitive_type
    ) or primitive_type_is_unsigned_integer(primitive_type)


def primitive_type_is_floating_point(primitive_type: PrimitiveType) -> bool:
    """Returns True iff the given SDFG primitive type is floating point"""
    return primitive_type in [
        PrimitiveType.Half,
        PrimitiveType.BFloat,
        PrimitiveType.Float,
        PrimitiveType.Double,
        PrimitiveType.X86_FP80,
        PrimitiveType.FP128,
        PrimitiveType.PPC_FP128,
    ]


TensorName = str
ContainerName = str


class ContainerMemory(Enum):
    """
    Enum for how the container memory is handled.
    UNMANGED: Nothing special is done. Default.
    MANAGED: The GraphParser manages the memory and determines when a malloc/free should be
             performed.
    IN_ARGUMENT: No memory management should be performed. The container is an input argument.
    OUT_ARGUMENT: No memory management should be performed. The container is an output argument.
    """

    UNMANAGED = 0
    MANAGED = 1
    IN_ARGUMENT = 2
    OUT_ARGUMENT = 3


class TensorInfo:
    """This class holds all information needed about a tensor"""

    _name: TensorName
    _sdfg_tensor_type: Tensor | None
    _container: ContainerName | None
    _depends_on: set[TensorName]
    _depended_by: set[TensorName]

    def __init__(
        self,
        name: TensorName,
        sdfg_tensor_type: Tensor | None = None,
        container: ContainerName | None = None,
    ) -> None:
        """Initialization"""
        self._name: TensorName = name
        self._sdfg_tensor_type: Tensor | None = sdfg_tensor_type
        self._container: ContainerName | None = container
        self._depends_on: set[TensorName] = set()
        self._depended_by: set[TensorName] = set()

    def name(self) -> TensorName:
        """Returns the tensor name"""
        return self._name

    def has_sdfg_tensor_type(self) -> bool:
        """True iff the tensor has a sdfg tensor type"""
        return not self._sdfg_tensor_type is None

    def sdfg_tensor_type(self) -> Tensor:
        """Returns the sdfg tensor type"""
        if self._sdfg_tensor_type is None:
            raise ValueError(
                "TensorInfo: Cannot access sdfg_tensor_info because its None"
            )
        return self._sdfg_tensor_type

    def set_sdfg_tensor_type(self, sdfg_tensor_type: Tensor | None) -> None:
        """Sets the SDFG tensor type of this tensor information to the provided SDFG tensor type"""
        self._sdfg_tensor_type = sdfg_tensor_type

    def element_type(self) -> Scalar:
        """Returns the element type of the sdfg tensor type"""
        return self.sdfg_tensor_type().element_type

    def shape(self) -> list[str]:
        """Returns the shape of the sdfg tensor type"""
        return self.sdfg_tensor_type().shape

    def shape_str(self) -> str:
        """
        Returns the shape of the sdfg tensor type as a string. This is needed for the out argument
        metadata.
        """
        return "[" + ",".join(self.sdfg_tensor_type().shape) + "]"

    def strides(self) -> list[str]:
        """Returns the strides of the sdfg tensor type"""
        return self.sdfg_tensor_type().strides

    def offset(self) -> str:
        """Returns the offset of the sdfg tensor type"""
        return self.sdfg_tensor_type().offset

    def is_contiguous(self) -> bool:
        """True iff the tensor is contiguous, i.e., has C-strides"""
        return self.sdfg_tensor_type().is_contiguous()

    def sdfg_type(self) -> Type:
        """Constructs an SDFG type for the underlying container"""
        if len(self.shape()) == 0:
            return self.element_type()
        else:
            return Pointer(self.element_type())

    def has_container(self) -> bool:
        """True iff the tensot info has an underlying container"""
        return not self._container is None

    def container(self) -> ContainerName:
        """Returns the underlying SDFG container to the tensor"""
        if self._container is None:
            raise ValueError("TensorInfo: Cannot access container because its None")
        return self._container

    def set_container(self, name: ContainerName) -> None:
        """Sets the container name of this tensor information to the provided name"""
        self._container: ContainerName | None = name

    def depends_on(self, name: TensorName) -> bool:
        """True iff this tensor depends on the provided tensor"""
        return name in self._depends_on

    def add_depends_on(self, name: TensorName) -> None:
        """Adds a dependency from this tensor the provided tensor"""
        self._depends_on.add(name)

    def depended_by(self, name: TensorName) -> bool:
        """True iff the provided tensor depends on this tensor"""
        return name in self._depended_by

    def add_depended_by(self, name: TensorName) -> None:
        """Adds a dependency from the provided tensor to this tensor"""
        self._depended_by.add(name)

    def __str__(self) -> str:
        """Prints the tensor information as a string. Helpful for debugging purposes."""
        stream: list[str] = [
            "TensorInfo('",
            self._name,
            "', ",
            str(self._sdfg_tensor_type),
            ", ",
        ]
        if self._container is None:
            stream.append(str(self._container))
        else:
            stream.append("'" + self._container + "'")
        stream.append(", {")
        stream.append(",".join({"'" + tensor + "'" for tensor in self._depends_on}))
        stream.append("}, {")
        stream.append(",".join({"'" + tensor + "'" for tensor in self._depended_by}))
        stream.append("})")
        return "".join(stream)


class TensorConstant:
    """This class holds all information needed about a tensor constant"""

    _value: str
    _sdfg_scalar: Scalar

    def __init__(self, value: str, sdfg_scalar: Scalar) -> None:
        """Initialization"""
        self._value: str = value
        self._sdfg_scalar: Scalar = sdfg_scalar

    def value(self) -> str:
        """Returns the constant value"""
        return self._value

    def sdfg_scalar(self) -> Scalar:
        """Returns the constant SDFG Scalar type"""
        return self._sdfg_scalar

    def has_sdfg_tensor_type(self) -> bool:
        """Always returns True to be compatible with TensorInfo"""
        return True

    def sdfg_tensor_type(self) -> Tensor:
        """
        Returns the sdfg tensor type constructed with empty shape from the constant SDFG Scalar type
        """
        return Tensor(self._sdfg_scalar, [])

    def element_type(self) -> Scalar:
        """Returns the constant SDFG Scalar type"""
        return self._sdfg_scalar

    def shape(self) -> list[str]:
        """Always returns the empty shape to be compatible with TensorInfo"""
        return []

    def strides(self) -> list[str]:
        """Always returns the empty strides to be compatible with TensorInfo"""
        return []

    def offset(self) -> str:
        """Always returns a zero offset to be compatible with TensorInfo"""
        return "0"

    def is_contiguous(self) -> bool:
        """Always returns True to be compatible with TensorInfo"""
        return True

    def sdfg_type(self) -> Type:
        """Constructs an SDFG type for the underlying container"""
        return self._sdfg_scalar

    def has_container(self) -> bool:
        """Always returns True to be compatible with TensorInfo"""
        return True

    def container(self) -> ContainerName:
        """Returns the constant value to be compatible with TensorInfo"""
        return self._value

    def __str__(self) -> str:
        """Prints the tensor information as a string. Helpful for debugging purposes."""
        return (
            "TensorConstant('" + self._value + "', " + self._sdfg_scalar.print() + ")"
        )


class ContainerInfo:
    """This class holds all information needed about an SDFG container"""

    _name: ContainerName
    _sdfg_type: Type
    _memory: ContainerMemory

    def __init__(
        self,
        name: ContainerName,
        sdfg_type: Type,
        memory: ContainerMemory = ContainerMemory.UNMANAGED,
    ) -> None:
        """Initialization. By default the conatiner memory is unmanaged."""
        self._name: ContainerName = name
        self._sdfg_type: Type = sdfg_type
        self._memory: ContainerMemory = memory

    def name(self) -> ContainerName:
        """Returns the name of this container"""
        return self._name

    def sdfg_type(self) -> Type:
        """Returns the SDFG type of this container"""
        return self._sdfg_type

    def memory(self) -> ContainerMemory:
        """Returns the container memory of this container"""
        return self._memory

    def memory_unmanaged(self) -> bool:
        """True iff the container memory is unmanged"""
        return self._memory == ContainerMemory.UNMANAGED

    def memory_managed(self) -> bool:
        """True iff the container memory is managed"""
        return self._memory == ContainerMemory.MANAGED

    def in_argument(self) -> bool:
        """True iff the container is an input argument"""
        return self._memory == ContainerMemory.IN_ARGUMENT

    def out_argument(self) -> bool:
        """True iff the container is an output argument"""
        return self._memory == ContainerMemory.OUT_ARGUMENT

    def set_memory(self, memory: ContainerMemory) -> None:
        """Sets the container memory of this container to the provided value"""
        self._memory: ContainerMemory = memory

    def __str__(self) -> str:
        """Prints the container information as a string. Helpful for debugging purposes."""
        return (
            "ContainerInfo('"
            + self._name
            + "', "
            + str(self._sdfg_type)
            + ", "
            + self._memory.name
            + ")"
        )


class TensorMetadata:
    """
    Holds the tensor and container information to each known tensor and container. Also stores
    information about lifetime.
    """

    _tensors: dict[TensorName, TensorInfo]
    _tensor_tuples: dict[TensorName, list[TensorName]]
    _containers: dict[ContainerName, ContainerInfo]
    _alive: set[ContainerName]

    def __init__(self) -> None:
        """Initialization"""
        self._tensors: dict[TensorName, TensorInfo] = {}
        self._tensor_tuples: dict[TensorName, list[TensorName]] = {}
        self._containers: dict[ContainerName, ContainerInfo] = {}
        self._alive = set()

    def has_tensor(self, name: TensorName) -> bool:
        """True iff there is tensor information about the provided tensor name"""
        return name in self._tensors

    def tensor(self, name: TensorName) -> TensorInfo:
        """Returns the tensor information corresponding to the provided tensor name"""
        return self._tensors[name]

    def add_tensor(self, name: TensorName, info: TensorInfo) -> None:
        """Adds the provided tensor information corresponding to the provided tensor name"""
        self._tensors[name] = info

    def is_tensor_tuple(self, name: TensorName) -> bool:
        """True iff the provided tensor name corresponds to a tensor tuple"""
        return name in self._tensor_tuples

    def tensor_tuple(self, name: TensorName) -> list[TensorName]:
        """Returns the list of tensor names that are the elements of the tensor tuple"""
        return self._tensor_tuples[name]

    def add_tensor_tuple(
        self, name: TensorName, contained_names: list[TensorName]
    ) -> None:
        """
        Adds the provided names contained in the tuple corresponding to the provided tensor name.
        """
        self._tensor_tuples[name] = contained_names

    def add_dependency(self, src: TensorName, dst: TensorName) -> None:
        """Add a tensor information dependency"""
        if self.is_tensor_tuple(src):
            for name in self.tensor_tuple(src):
                self._tensors[name].add_depends_on(dst)
        else:
            self._tensors[src].add_depends_on(dst)
        if self.is_tensor_tuple(dst):
            for name in self.tensor_tuple(dst):
                self._tensors[name].add_depended_by(src)
        else:
            self._tensors[dst].add_depended_by(src)

    def has_container(self, name: ContainerName) -> bool:
        """True iff there is container information about the provided container name"""
        return name in self._containers

    def container(self, name: ContainerName) -> ContainerInfo:
        """Returns the container information corresponding to the provided container name"""
        return self._containers[name]

    def add_container(self, name: ContainerName, info: ContainerInfo) -> None:
        """Adds the provided container information corresponding to the provided container name"""
        self._containers[name] = info

    def is_alive(self, name: ContainerName) -> bool:
        """True iff the provided container is alive"""
        return name in self._alive

    def live(self, name: ContainerName) -> None:
        """Sets the provided container to be alive"""
        self._alive.add(name)

    def dead(self, name: ContainerName) -> None:
        """Sets the provided container to be dead"""
        self._alive.remove(name)

    def containers_memory_managed(self) -> list[ContainerName]:
        """Returns a list of container names for all memory managed containers"""
        return [
            name for name, info in self._containers.items() if info.memory_managed()
        ]

    def __str__(self) -> str:
        """Prints the whole tensor metadata as a string. Helpful for debugging purposes."""
        stream: list[str] = ["TensorMetadata(\n  tensors = {\n"]
        for name, info in self._tensors.items():
            stream.extend(["    '", name, "': ", str(info), ",\n"])
        stream.append("  },\n  tensor_tuples = {\n")
        for name, contained_names in self._tensor_tuples.items():
            stream.extend(["    '", name, "': ["])
            stream.append(
                ",".join(
                    ["'" + contained_name + "'" for contained_name in contained_names]
                )
            )
            stream.append("],\n")
        stream.append("  },\n  containers = {\n")
        for name, info in self._containers.items():
            stream.extend(["    '", name, "': ", str(info), ",\n"])
        stream.append("  },\n  alive = {")
        stream.append(",".join({"'" + name + "'" for name in self._alive}))
        stream.append("}\n)\n")
        return "".join(stream)


class GraphParserErrorBase(Exception):
    """Custom exception that prints PyTorch stack trace if available"""

    def __init__(self, node: torch.fx.Node, message: str) -> None:
        passed_message: str = message
        if "stack_trace" in node.meta:
            passed_message += "\nStack trace:\n" + node.meta["stack_trace"]
        super().__init__(passed_message)


class GraphParserError(GraphParserErrorBase):
    """Custom exception that prints current class and PyTorch stack trace if available"""

    def __init__(self, current: object, node: torch.fx.Node, message: str) -> None:
        super().__init__(node, current.__class__.__name__ + ": " + message)


TORCH_PRIMITIVE_TYPES: dict[torch.dtype, PrimitiveType] = {
    torch.float32: PrimitiveType.Float,
    torch.float: PrimitiveType.Float,
    torch.float64: PrimitiveType.Double,
    torch.double: PrimitiveType.Double,
    torch.float16: PrimitiveType.Half,
    torch.half: PrimitiveType.Half,
    torch.bfloat16: PrimitiveType.BFloat,
    # Unsupported: torch.complex32
    # Unsupported: torch.chalf
    # Unsupported: torch.complex64
    # Unsupported: torch.cfloat
    # Unsupported: torch.complex128
    # Unsupported: torch.cdouble
    # Unsupported: torch.float8_e4m3fn
    # Unsupported: torch.float8_e5m2
    # Unsupported: torch.float8_e4m3fnuz
    # Unsupported: torch.float8_e5m2fnuz
    # Unsupported: torch.float8_e8m0fnuz
    # Unsupported: torch.float8_e2m1fn_x2
    torch.uint8: PrimitiveType.UInt8,
    torch.int8: PrimitiveType.Int8,
    torch.uint16: PrimitiveType.UInt16,
    torch.int16: PrimitiveType.Int16,
    torch.short: PrimitiveType.Int16,
    torch.uint32: PrimitiveType.UInt32,
    torch.int32: PrimitiveType.Int32,
    torch.int: PrimitiveType.Int32,
    torch.uint64: PrimitiveType.UInt64,
    torch.int64: PrimitiveType.Int64,
    torch.long: PrimitiveType.Int64,
    torch.bool: PrimitiveType.Bool,
}


class GraphParserBase:
    """
    Base class for everything in the PyTorch GraphModule Parser. Contains helper method for
    converting PyTorch types to SDFG types, PyTorch nodes to SDFG containers, and PyTorch stack
    traces to SDFG debug information.
    """

    def determine_sdfg_scalar_type(self, node: torch.fx.Node, input: Any) -> Scalar:
        """
        Tries to convert a PyTorch type to an SDFG Scalar type. If it fails, an exception is thrown.
        """
        if isinstance(input, int):
            return Scalar(PrimitiveType.Int64)
        elif isinstance(input, float):
            return Scalar(PrimitiveType.Double)
        elif isinstance(input, bool):
            return Scalar(PrimitiveType.Bool)
        elif isinstance(input, torch.dtype):
            if not input in TORCH_PRIMITIVE_TYPES:
                raise GraphParserError(
                    self, node, f"No primitive sdfg type for torch.dtype: {input}"
                )
            return Scalar(TORCH_PRIMITIVE_TYPES[input])
        raise GraphParserError(self, node, f"Unknown type: {type(input)}")

    def determine_sdfg_type(self, node: torch.fx.Node, input: Any) -> Type:
        """
        Tries to convert a PyTorch type to an SDFG type. If the conversion fails, an exception is
        thrown.
        """
        if isinstance(input, torch.Tensor):
            base_type: Scalar = self.determine_sdfg_scalar_type(node, input.dtype)
            if len(input.shape) == 0:
                return base_type
            else:
                return Pointer(base_type)
        # Fallback to scalar types
        return self.determine_sdfg_scalar_type(node, input)

    def get_node_sdfg_type(self, node: torch.fx.Node) -> Type:
        """
        Tries to converts the return type of a PyTorch node to an SDFG type. If the conversion
        fails, an exception is thrown.
        """
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        return self.determine_sdfg_type(node, node.meta["val"])

    def determine_sdfg_tensor_type(
        self, node: torch.fx.Node, input: Any
    ) -> Tensor | None:
        """Tries to convert a PyTorch type to an SDFG tensor type. If it fails, None is returned."""
        if isinstance(input, torch.Tensor):
            base_type: Scalar = self.determine_sdfg_scalar_type(node, input.dtype)
            tensor_shape: list[str] = [str(dim) for dim in input.shape]
            tensor_stride: list[str] = [str(stride) for stride in input.stride()]
            tensor_offset: str = str(input.storage_offset())
            return Tensor(base_type, tensor_shape, tensor_stride, tensor_offset)
        return None

    def update_with_tensor_meta(
        self,
        node: torch.fx.Node,
        tensor: Tensor,
        tensor_meta: torch.fx.passes.shape_prop.TensorMetadata,
    ) -> Tensor:
        """
        Update an SDFG tensor type with a PyTorch TensorMetadata structure and returns the updated
        SDFG tensor type.
        """
        base_type: Scalar = self.determine_sdfg_scalar_type(node, tensor_meta.dtype)
        tensor_shape: list[str] = [str(dim) for dim in tensor_meta.shape]
        tensor_stride: list[str] = [str(stride) for stride in tensor_meta.stride]
        if tensor.element_type.primitive_type != base_type.primitive_type:
            raise GraphParserError(
                self,
                node,
                f"Tensor and TensorMetadata element type mismatch: {tensor.element_type.print()} != {base_type.print()}",
            )
        if tensor.shape != tensor_shape:
            raise GraphParserError(
                self,
                node,
                f"Tensor and TensorMetadata shape mismatch: {tensor.shape} != {tensor_shape}",
            )
        if tensor.strides != tensor_stride:
            raise GraphParserError(
                self,
                node,
                f"Tensor and TensorMetadata strides mismatch: {tensor.strides} != {tensor_stride}",
            )
        return tensor

    def get_node_sdfg_tensor(self, node: torch.fx.Node) -> Tensor | None:
        """
        Tries to converts the return type of a PyTorch node to an SDFG tensor type. If the
        conversion fails, None is returned. Calls ``determine_sdfg_tensor_type`` and
        ``update_with_tensor_meta`` in the process. For a PyTorch tuple return type, use
        ``get_node_sdfg_tensors``.
        """
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        val: Any = node.meta["val"]
        sdfg_tensor: Tensor | None = self.determine_sdfg_tensor_type(node, val)
        if not sdfg_tensor is None and "tensor_meta" in node.meta:
            tensor_meta: Any = node.meta["tensor_meta"]
            if not isinstance(tensor_meta, torch.fx.passes.shape_prop.TensorMetadata):
                raise GraphParserError(
                    self,
                    node,
                    "Expected tensor_meta metadata to be TensorMetadata type but got: "
                    + str(type(tensor_meta)),
                )
            sdfg_tensor: Tensor | None = self.update_with_tensor_meta(
                node, sdfg_tensor, tensor_meta
            )
        return sdfg_tensor

    def get_node_sdfg_tensors(self, node: torch.fx.Node) -> tuple[Tensor | None, ...]:
        """
        Tries to converts the tuple return type of a PyTorch node to an SDFG tensor types. For each
        element for which the conversion fails, None is returned. Calls
        ``determine_sdfg_tensor_type`` in the process. For a PyTorch non-tuple return type, use
        ``get_node_sdfg_tensor``.
        """
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        val: Any = node.meta["val"]
        if not isinstance(val, tuple):
            return (self.get_node_sdfg_tensor(node),)
        sdfg_tensors: list[Tensor | None] = []
        for elem in val:
            sdfg_tensors.append(self.determine_sdfg_tensor_type(node, elem))
        return tuple(sdfg_tensors)

    def convert_arg_to_tensor_info(
        self,
        node: torch.fx.Node,
        metadata: TensorMetadata,
        arg: Argument,
    ) -> TensorInfo:
        """
        Tries to convert a PyTorch Argument to an SDFG tensor information. If it fails, an exception
        is thrown.
        """
        if not isinstance(arg, torch.fx.Node):
            raise GraphParserError(
                self, node, f"Cannot convert argument to tensor info: {type(arg)}"
            )
        name: TensorName = arg.name
        if not metadata.has_tensor(name):
            raise GraphParserError(
                self, node, f"No tensor information available: " + name
            )
        return metadata.tensor(name)

    def convert_arg_to_tensor_constant(
        self, node: torch.fx.Node, arg: Argument
    ) -> TensorConstant:
        """
        Tries to convert a PyTorch Argument to a tensor constant. If it fails, an exception is
        thrown.
        """
        if isinstance(arg, (int, float, bool)):
            constant_scalar: Scalar = self.determine_sdfg_scalar_type(node, arg)
            if isinstance(arg, float):
                if math.isnan(arg):
                    return TensorConstant("NAN", constant_scalar)
                elif arg == math.inf:
                    return TensorConstant("INFINITY", constant_scalar)
                elif arg == -math.inf:
                    return TensorConstant("-INFINITY", constant_scalar)
            return TensorConstant(str(arg), constant_scalar)
        raise GraphParserError(
            self, node, f"Cannot convert argument to tensor constant: {type(arg)}"
        )

    def align_constant_type(
        self, node: torch.fx.Node, constant: TensorConstant, dst_type: Scalar
    ) -> Scalar:
        """
        Align a tensor constant to a Scalar destination type. This is helpful for avoiding
        unnecessary casts. For example, if a tensor with base type float should be scaled by a
        double or integer constant elementwisely, this method would convert it to float.
        """
        constant_prim: PrimitiveType = constant.sdfg_scalar().primitive_type
        if primitive_type_is_integer(dst_type.primitive_type):
            limits: dict[PrimitiveType, tuple[int, int]] = {
                PrimitiveType.Bool: (0, 1),
                PrimitiveType.Int8: (-128, 127),
                PrimitiveType.Int16: (-32_768, 32_767),
                PrimitiveType.Int32: (-2_147_483_648, 2_147_483_647),
                PrimitiveType.Int64: (
                    -9_223_372_036_854_775_808,
                    9_223_372_036_854_775_807,
                ),
                PrimitiveType.Int128: (
                    -170_141_183_460_469_231_731_687_303_715_884_105_728,
                    170_141_183_460_469_231_731_687_303_715_884_105_727,
                ),
                PrimitiveType.UInt8: (0, 255),
                PrimitiveType.UInt16: (0, 65_535),
                PrimitiveType.UInt32: (0, 4_294_967_295),
                PrimitiveType.UInt64: (0, 18_446_744_073_709_551_615),
                PrimitiveType.UInt128: (
                    0,
                    340_282_366_920_938_463_463_374_607_431_768_211_455,
                ),
            }
            limit: tuple[int, int] = limits[dst_type.primitive_type]
            if primitive_type_is_integer(constant_prim):
                int_val: int = int(constant.value())
                if int_val >= limit[0] and int_val <= limit[1]:
                    return dst_type
        elif primitive_type_is_floating_point(dst_type.primitive_type):
            if primitive_type_is_integer(
                constant_prim
            ) or primitive_type_is_floating_point(constant_prim):
                return dst_type
        raise GraphParserError(
            self,
            node,
            f"Cannot align constant type: {constant_prim} -> {dst_type.primitive_type}",
        )

    def convert_arg_to_tensor_info_or_constant(
        self,
        node: torch.fx.Node,
        metadata: TensorMetadata,
        arg: Argument,
        align_constant_type: Scalar | None = None,
    ) -> TensorInfo | TensorConstant:
        """
        Tries to convert a PyTorch Argument to either a tensor info or a tensor constant. If both
        fails, an exception is thrown.
        """
        if isinstance(arg, torch.fx.Node):
            return self.convert_arg_to_tensor_info(node, metadata, arg)
        elif isinstance(arg, (int, float, bool)):
            constant: TensorConstant = self.convert_arg_to_tensor_constant(node, arg)
            if align_constant_type is None:
                return constant
            else:
                new_constant_scalar: Scalar = self.align_constant_type(
                    node, constant, align_constant_type
                )
                return TensorConstant(constant.value(), new_constant_scalar)
        raise GraphParserError(
            self,
            node,
            f"Cannot convert argument to neither tensor info nor tensor constant: {type(arg)}",
        )

    def convert_arg_to_expr(self, node: torch.fx.Node, arg: Argument) -> str:
        """
        Tries to convert a PyTorch Argument to an SDFG symbolic expression. If it fails, an
        exception is thrown.
        """
        if isinstance(arg, int):
            return str(arg)
        raise GraphParserError(
            self, node, f"Cannot convert argument to symbolic expression: {type(arg)}"
        )

    def convert_arg_to_multi_expr(
        self, node: torch.fx.Node, arg: Argument
    ) -> list[str]:
        """
        Tries to convert a PyTorch Argument to an SDFG symbolic multi expression (list of symbolic
        expressions). If it fails, an exception is thrown.
        """
        if isinstance(arg, list):
            return [self.convert_arg_to_expr(node, elem) for elem in arg]
        raise GraphParserError(
            self,
            node,
            f"Cannot convert argument to symbolic multi expression: {type(arg)}",
        )

    def parse_torch_2_13_0_stack_trace(self, stack_trace: str) -> DebugInfo:
        """
        Parses a PyTorch (version 2.13.0 and newer) stack trace to SDFG debug information. If the
        parsing fails, an empty debug information is returned.
        """
        if len(stack_trace.strip()) == 0:
            return DebugInfo()
        lines: list[str] = stack_trace.split("\n")
        if len(lines) == 0:
            return DebugInfo()
        if len(lines[-1]) == 0:
            lines.pop()
        if len(lines) == 0:
            return DebugInfo()
        last_line_chars = set(list(lines[-1].strip()))
        if len(last_line_chars) == 1 and "^" in last_line_chars:
            lines.pop()
        if len(lines) < 2:
            return DebugInfo()
        line: str = lines[-2].strip()
        parts: list[str] = line.split(", ")
        if len(parts) != 3:
            return DebugInfo()
        filename: str = ""
        function: str = ""
        start_line: int = 0
        if parts[0].startswith('File "') and parts[0].endswith('"'):
            filename = parts[0][6:-1]
        if parts[1].startswith("line ") and parts[1][5:].isnumeric():
            start_line = int(parts[1][5:])
        if parts[2].startswith("in "):
            function = parts[2][3:]
        end_col: int = len(lines[-1].strip())
        return DebugInfo(filename, function, start_line, 0, start_line, end_col)

    def parse_torch_stack_trace(self, stack_trace: str) -> DebugInfo:
        """
        Parses a PyTorch stack trace to SDFG debug information. If the parsing fails, an empty
        debug information is returned.
        """
        from torch.torch_version import TorchVersion

        # PyTorch 2.13.0+ uses a slightly different format
        if torch.__version__ >= TorchVersion("2.13.0"):
            return self.parse_torch_2_13_0_stack_trace(stack_trace)

        if len(stack_trace.strip()) == 0:
            return DebugInfo()
        lines: list[str] = stack_trace.split("\n")
        if len(lines) == 0:
            return DebugInfo()
        if len(lines[-1]) == 0:
            lines.pop()
        if len(lines) < 2:
            return DebugInfo()
        line: str = lines[-2].strip()
        parts: list[str] = line.split(", ")
        if len(parts) != 3:
            return DebugInfo()
        filename: str = ""
        function: str = ""
        start_line: int = 0
        if parts[0].startswith('File "') and parts[0].endswith('"'):
            filename = parts[0][6:-1]
        if parts[1].startswith("line ") and parts[1][5:].isnumeric():
            start_line = int(parts[1][5:])
        if parts[2].startswith("in "):
            function = parts[2][3:]
        end_col: int = len(lines[-1].strip())
        return DebugInfo(filename, function, start_line, 0, start_line, end_col)

    def get_debug_info(self, node: torch.fx.Node) -> DebugInfo:
        """
        Converts the PyTorch stack trace attached to the node to SDFG debug information if available
        """
        return self.parse_torch_stack_trace(
            "" if not "stack_trace" in node.meta else node.meta["stack_trace"]
        )

    def copy_output_tensor(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        src_info: TensorInfo,
        dst_info: TensorInfo,
    ) -> None:
        """
        This is a helper function to copy output tensors. This can happen:
        - if the output tensor is non-contiguous because we only allow output tensors with
          C-strides,
        - if the output tensor is scalar because we cannot have a scalar output argument.
        """
        if not src_info.has_sdfg_tensor_type():
            raise GraphParserError(
                self,
                node,
                "Cannot copy from src tensor because it has no SDFG tensor type: "
                + str(src_info),
            )
        if not dst_info.has_sdfg_tensor_type():
            raise GraphParserError(
                self,
                node,
                "Cannot copy to dst tensor because it has no SDFG tensor type: "
                + str(dst_info),
            )
        if not src_info.has_container():
            raise GraphParserError(
                self,
                node,
                "Cannot copy from src tensor because it has no SDFG container: "
                + str(src_info),
            )
        if not dst_info.has_container():
            raise GraphParserError(
                self,
                node,
                "Cannot copy to dst tensor because it has no SDFG container: "
                + str(dst_info),
            )

        debug_info: DebugInfo = self.get_debug_info(node)
        if not src_info.is_contiguous() and dst_info.is_contiguous():
            # This means that the model would return a non-contiguous tensor. We need to perform a
            # copy to make it contiguous.
            builder.add_copy_op(
                src_info.container(),
                src_info.sdfg_tensor_type(),
                dst_info.container(),
                dst_info.sdfg_tensor_type(),
                debug_info,
            )
        elif src_info.shape() == [] and dst_info.shape() == ["1"]:
            # This means that the model would return a scalar tensor, i.e., a scalar. However, we
            # can only return pointers. So, a copy is performed.
            block: Block = builder.add_block(debug_info)
            src_access: AccessNode = builder.add_access(
                block, src_info.container(), debug_info
            )
            dst_access: AccessNode = builder.add_access(
                block, dst_info.container(), debug_info
            )
            tasklet: Tasklet = builder.add_tasklet(
                block, TaskletCode.assign, ["_in"], ["_out"], debug_info
            )
            builder.add_memlet(
                block,
                src_access,
                "void",
                tasklet,
                "_in",
                debug_info=debug_info,
            )
            builder.add_memlet(
                block,
                tasklet,
                "_out",
                dst_access,
                "void",
                subset="0",
                debug_info=debug_info,
            )
        else:
            raise GraphParserError(
                self,
                node,
                "Cannot copy output tensors: " + str(src_info) + " -> " + str(dst_info),
            )


class GraphParserModule(GraphParserBase, ABC):
    """
    This is the base class for a module in the PyTorch GraphModule Parser. For each operation a
    GraphParser module can be registered. The job of the module is to parse only its registered
    operations. This base class provides all the helper and utility function needed for that.
    """

    @abstractmethod
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> None:
        """
        This function is called from the GraphParser to dispatch to the GraphParser module. Its
        purpose is to translate the provided operation into an equivalent SDFG operation.
        """
        pass

    def get_arg_tensor_info(
        self,
        node: torch.fx.Node,
        metadata: TensorMetadata,
        index: int,
        check_tensor_present: bool = True,
        check_container_present: bool = True,
    ) -> TensorInfo:
        """
        Convert the index-th PyTorch Argument to a tensor information. Throws an exception if the
        index is out of bounds. The flag check_tensor_present checks if the tensor information has
        an SDFG tensor type set. The flag check_container_present checks that the tensor information
        already has an SDFG container. Those are done by default. See ``convert_arg_to_tensor_info``
        for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        info: TensorInfo = self.convert_arg_to_tensor_info(
            node, metadata, node.args[index]
        )
        if check_tensor_present and not info.has_sdfg_tensor_type():
            raise GraphParserError(
                self,
                node,
                f"Expected an SDFG tensor type to be present for the {index+1}. argument",
            )
        if check_container_present and not info.has_container():
            raise GraphParserError(
                self,
                node,
                f"Expected an SDFG container to be present for the {index+1}. argument",
            )
        return info

    def get_arg_tensor_info_or_constant(
        self,
        node: torch.fx.Node,
        metadata: TensorMetadata,
        index: int,
        align_constant_type: Scalar | None = None,
        check_tensor_present: bool = True,
        check_container_present: bool = True,
    ) -> TensorInfo | TensorConstant:
        """
        Convert the index-th PyTorch Argument to either a tensor information or a tensor constant.
        Throws an exception if the index is out of bounds. If the align_constant_type flag is set to
        a SDFG scalar, ``align_constant_type`` is called on the tensor constant (if applicable). See
        ``get_arg_tensor_info`` for more information about the other flags. Also see
        ``convert_arg_to_tensor_info_or_constant`` for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        info: TensorInfo | TensorConstant = self.convert_arg_to_tensor_info_or_constant(
            node, metadata, node.args[index], align_constant_type=align_constant_type
        )
        if check_tensor_present and not info.has_sdfg_tensor_type():
            raise GraphParserError(
                self,
                node,
                f"Expected an SDFG tensor type to be present for the {index+1}. argument",
            )
        if check_container_present and not info.has_container():
            raise GraphParserError(
                self,
                node,
                f"Expected an SDFG container to be present for the {index+1}. argument",
            )
        return info

    def get_arg_expr(self, node: torch.fx.Node, index: int) -> str:
        """
        Convert the index-th PyTorch Argument to an SDFG symbolic expression. Throws an exception if
        the index is out of bounds. See ``convert_arg_to_expr`` for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        return self.convert_arg_to_expr(node, node.args[index])

    def get_arg_multi_expr(self, node: torch.fx.Node, index: int) -> list[str]:
        """
        Convert the index-th PyTorch Argument to an SDFG symbolic multi expression (list of symbolic
        expressions). Throws an exception if the index is out of bounds. See
        ``convert_arg_to_multi_expr`` for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        return self.convert_arg_to_multi_expr(node, node.args[index])

    def allocate_memory(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        tensor_info: TensorInfo,
        debug_info: DebugInfo | None = None,
    ) -> None:
        """
        Adds a memory allocation (malloc) to the SDFG for the underlying container of the tensor
        information. The size is obtained from the SDFG tensor type. If the size is 0 (Scalar
        Tensor), this function is a NOP.
        """
        if debug_info is None:
            debug_info_: DebugInfo = self.get_debug_info(node)
        else:
            debug_info_: DebugInfo = debug_info

        if not tensor_info.has_sdfg_tensor_type():
            raise GraphParserError(
                self,
                node,
                "Could not allocate memory for non-tensor-typed tensor: "
                + tensor_info.name(),
            )
        if not tensor_info.has_container():
            raise GraphParserError(
                self,
                node,
                "Could not allocate memory for tensor without container: "
                + tensor_info.name(),
            )
        size: str = tensor_info.sdfg_tensor_type().total_size()
        if size != "0":
            builder.add_malloc_block(tensor_info.container(), size, debug_info_)

    def get_result_tensor_info(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> TensorInfo:
        """
        Returns (and potentially creates) a tensor information to use as the result of the current
        operation. For creating a tensor information for an intermediate result, see
        ``create_intermediate_tensor_info``. If the operation has multiple results (tuple), use
        ``create_result_tensor_infos``.
        """
        if metadata.has_tensor(node.name):
            tensor_info: TensorInfo = metadata.tensor(node.name)
        else:
            sdfg_tensor: Tensor | None = self.get_node_sdfg_tensor(node)
            tensor_info: TensorInfo = TensorInfo(node.name, sdfg_tensor)
            metadata.add_tensor(node.name, tensor_info)

        if tensor_info.has_container():
            container_info: ContainerInfo = metadata.container(tensor_info.container())
        elif metadata.has_container(node.name):
            container_info: ContainerInfo = metadata.container(node.name)
            tensor_info.set_container(node.name)
        else:
            sdfg_type: Type = self.get_node_sdfg_type(node)
            builder.add_container(node.name, sdfg_type)
            container_info: ContainerInfo = ContainerInfo(
                node.name, sdfg_type, ContainerMemory.MANAGED
            )
            metadata.add_container(node.name, container_info)
            tensor_info.set_container(node.name)

        if container_info.memory_managed() and isinstance(
            container_info.sdfg_type(), Pointer
        ):
            self.allocate_memory(node, builder, metadata, tensor_info)

        metadata.live(container_info.name())

        return tensor_info

    def get_result_tensor_infos(
        self,
        num: int,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
    ) -> tuple[TensorInfo, ...]:
        """
        Returns (and potentially creates) tensor information to use as the results of the current
        operation. Because SDFGs do not support tuples, a tensor information for each element of the
        tuple is created. Access to them are resolved with the help of virtual tensor information.
        For creating only a single result of the current operation, see ``get_result_tensor_info``.
        """
        sdfg_tensors: tuple[Tensor | None, ...] = self.get_node_sdfg_tensors(node)
        if len(sdfg_tensors) != num:
            raise GraphParserError(
                self,
                node,
                f"Expected {num} result tensors but got: {len(sdfg_tensors)}",
            )

        infos: list[TensorInfo] = []
        for i in range(num):
            name: TensorName = f"{node.name}_{i}"
            if metadata.has_tensor(name):
                tensor_info: TensorInfo = metadata.tensor(name)
            else:
                tensor_info: TensorInfo = TensorInfo(name, sdfg_tensors[i])
                metadata.add_tensor(name, tensor_info)

            if tensor_info.has_container():
                container_info: ContainerInfo = metadata.container(
                    tensor_info.container()
                )
            elif metadata.has_container(name):
                container_info: ContainerInfo = metadata.container(name)
                tensor_info.set_container(name)
            else:
                sdfg_type: Type = tensor_info.sdfg_type()
                builder.add_container(name, sdfg_type)
                container_info: ContainerInfo = ContainerInfo(
                    name, sdfg_type, ContainerMemory.MANAGED
                )
                metadata.add_container(name, container_info)
                tensor_info.set_container(name)

            if container_info.memory_managed() and isinstance(
                container_info.sdfg_type(), Pointer
            ):
                self.allocate_memory(node, builder, metadata, tensor_info)

            metadata.live(name)

            infos.append(tensor_info)

        return tuple(infos)

    def create_view(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        info: TensorInfo,
        ref_info: TensorInfo,
    ) -> None:
        """
        Create a view from one tensor information to another tensor information. That means, that
        the first tensor information will start using the container of the second tensor
        information. This is not done in the special case that the second tensor information has an
        underlying output argument container. In that case, a copy/broadcast is added.
        """
        if not ref_info.has_container():
            raise GraphParserError(
                self,
                node,
                "Cannot create view from tensor information without container: "
                + str(ref_info),
            )
        if info.has_container():
            if not info.has_sdfg_tensor_type():
                raise GraphParserError(
                    self,
                    node,
                    "Cannot create view (copy) from tensor information without SDFG tensor type: "
                    + str(info),
                )
            if not ref_info.has_sdfg_tensor_type():
                raise GraphParserError(
                    self,
                    node,
                    "Cannot create view (copy) from tensor information without SDFG tensor type: "
                    + str(ref_info),
                )
            debug_info: DebugInfo = self.get_debug_info(node)
            if (
                info.sdfg_tensor_type().total_elements()
                == ref_info.sdfg_tensor_type().total_elements()
            ):
                builder.add_copy_op(
                    ref_info.container(),
                    ref_info.sdfg_tensor_type(),
                    info.container(),
                    info.sdfg_tensor_type(),
                    debug_info,
                )
            else:
                builder.add_broadcast_op(
                    ref_info.container(),
                    ref_info.sdfg_tensor_type(),
                    info.container(),
                    info.sdfg_tensor_type(),
                    ref_info.shape(),
                    info.shape(),
                    debug_info,
                )
        else:
            info.set_container(ref_info.container())
        metadata.live(info.container())

    def create_result_view(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        ref_info: TensorInfo,
    ) -> None:
        """
        Create a view from the tensor information corresponding to the result of the node to the
        provided tensor information. See ``create_view`` for more information.
        """
        if metadata.has_tensor(node.name):
            info: TensorInfo = metadata.tensor(node.name)
        else:
            sdfg_tensor: Tensor | None = self.get_node_sdfg_tensor(node)
            info: TensorInfo = TensorInfo(node.name, sdfg_tensor)
            metadata.add_tensor(node.name, info)
        self.create_view(node, builder, metadata, info, ref_info)

    def create_intermediate_tensor_info(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        metadata: TensorMetadata,
        sdfg_type: Type,
        sdfg_tensor_type: Tensor | None = None,
        dependencies: list[TensorName] | list[TensorInfo | TensorConstant] = [],
    ) -> TensorInfo:
        """
        Creates a tensor information with corresponding container information to use for
        intermediate results. Memory allocation and management is automatically handled. DO NOT use
        this method for creating result information. See ``get_result_tensor_info`` and
        ``get_result_tensor_infos`` for that.
        """
        name: str = builder.find_new_name("intermediate")
        builder.add_container(name, sdfg_type)
        metadata.add_container(
            name, ContainerInfo(name, sdfg_type, ContainerMemory.MANAGED)
        )
        info: TensorInfo = TensorInfo(name, sdfg_tensor_type, name)
        metadata.add_tensor(name, info)
        if isinstance(sdfg_type, Pointer):
            self.allocate_memory(node, builder, metadata, info)
        for dependency in dependencies:
            if isinstance(dependency, TensorName):
                metadata.add_dependency(name, dependency)
            elif isinstance(dependency, TensorInfo):
                metadata.add_dependency(name, dependency.name())
        metadata.live(info.container())
        return info

    def align_elementwise_tensors(
        self, node: torch.fx.Node, tensor1: Tensor, tensor2: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Align two tensor types to each other, i.e., virtually broadcast the tensor type with smaller
        shape to the tensor type with bigger shape by adding zero stride entries. For example:
        align_elementwise_tensors(Tensor(shape=[3],strides=(1,)), Tensor(shape=[2,3],strides=(3,1)))
         = Tensor(shape=[2,3],strides=(0,1)), Tensor(shape=[2,3],strides=(3,1))
        """
        dims1: int = len(tensor1.shape)
        dims2: int = len(tensor2.shape)
        if dims1 == dims2 or dims1 == 0 or dims2 == 0:
            return tensor1, tensor2

        if dims1 < dims2:
            if tensor2.shape[-dims1:] != tensor1.shape:
                raise GraphParserError(
                    self,
                    node,
                    "Cannot align elementwise tensors: "
                    + tensor1.print()
                    + " and "
                    + tensor2.print(),
                )
            new_strides: list[str] = [
                "0" for _ in range(dims2 - dims1)
            ] + tensor1.strides
            return (
                Tensor(
                    tensor1.element_type, tensor2.shape, new_strides, tensor1.offset
                ),
                tensor2,
            )
        else:  # dims2 > dims1
            if tensor1.shape[-dims2:] != tensor2.shape:
                raise GraphParserError(
                    self,
                    node,
                    "Cannot align elementwise tensors: "
                    + tensor1.print()
                    + " and "
                    + tensor2.print(),
                )
            new_strides: list[str] = [
                "0" for _ in range(dims1 - dims2)
            ] + tensor2.strides
            return tensor1, Tensor(
                tensor2.element_type, tensor1.shape, new_strides, tensor2.offset
            )

    def get_kwarg_dtype(self, node: torch.fx.Node) -> torch.dtype | None:
        """Check and return the "dtype" kwarg as torch.dtype if available"""
        if not "dtype" in node.kwargs:
            return
        dtype_arg: Argument = node.kwargs["dtype"]
        if dtype_arg is None:
            return
        if not isinstance(dtype_arg, torch.dtype):
            raise GraphParserError(
                self,
                node,
                "Expected dtype kwarg to be torch.dtype type but got: "
                + str(type(dtype_arg)),
            )
        return dtype_arg

    def get_kwarg_layout(self, node: torch.fx.Node) -> torch.layout | None:
        """Check and return the "layout" kwarg as torch.layout if available"""
        if not "layout" in node.kwargs:
            return
        layout_arg: Argument = node.kwargs["layout"]
        if layout_arg is None:
            return
        if not isinstance(layout_arg, torch.layout):
            raise GraphParserError(
                self,
                node,
                "Expected layout kwarg to be torch.layout type but got: "
                + str(type(layout_arg)),
            )
        if layout_arg != torch.strided:
            raise GraphParserError(
                self,
                node,
                "Only layout torch.strided is supported but got: " + str(layout_arg),
            )
        return layout_arg

    def get_kwarg_device(self, node: torch.fx.Node) -> torch.device | None:
        """Check and return the "device" kwarg as torch.device if available"""
        if not "device" in node.kwargs:
            return
        device_arg: Argument = node.kwargs["device"]
        if device_arg is None:
            return
        if not isinstance(device_arg, torch.device):
            raise GraphParserError(
                self,
                node,
                "Expected device kwarg to be torch.device type but got: "
                + str(type(device_arg)),
            )
        if device_arg.type != "cpu":
            raise GraphParserError(
                self, node, "Currently only CPU device kwarg is supported"
            )
        return device_arg

    def get_kwarg_pin_memory(self, node: torch.fx.Node) -> bool | None:
        """Check and return the "pin_memory" kwarg as bool if available"""
        if not "pin_memory" in node.kwargs:
            return
        pin_memory_arg: Argument = node.kwargs["pin_memory"]
        if pin_memory_arg is None:
            return
        if not isinstance(pin_memory_arg, bool):
            raise GraphParserError(
                self,
                node,
                "Expected pin_memory kwarg to be bool type but got: "
                + str(type(pin_memory_arg)),
            )
        if pin_memory_arg:
            raise GraphParserError(self, node, "Currently pin_memory is unsupported")
        return pin_memory_arg

    def get_kwarg_memory_format(
        self, node: torch.fx.Node
    ) -> torch.memory_format | None:
        """Check and return the "memory_format" kwarg as torch.memory_format if available"""
        if not "memory_format" in node.kwargs:
            return
        memory_format_arg: Argument = node.kwargs["memory_format"]
        if memory_format_arg is None:
            return
        if not isinstance(memory_format_arg, torch.memory_format):
            raise GraphParserError(
                self,
                node,
                "Expected memory_format kwarg to be torch.memory_format type but got: "
                + str(type(memory_format_arg)),
            )
        if memory_format_arg not in [torch.contiguous_format, torch.preserve_format]:
            raise GraphParserError(
                self, node, "Unsupported memory_format: " + str(memory_format_arg)
            )
        return memory_format_arg


GRAPH_PARSER_MODULES: dict[str, GraphParserModule] = {}


def get_node_target_name(target: Target) -> str:
    """Helper method to convert a PyTorch Target to a string"""
    if isinstance(target, str):
        return target
    elif target.__module__ == "torch._ops.aten":
        return str(target)
    else:
        return target.__module__ + "." + target.__name__


def register_module(op: str, module: GraphParserModule) -> None:
    """
    Registers a GraphParser module to an operation for the parsing step. Throws if another module is
    already registered to that operation.
    """
    if op in GRAPH_PARSER_MODULES:
        raise KeyError(
            f"GraphParser: Could not register module because it already exists: {op}"
        )
    GRAPH_PARSER_MODULES[op] = module


def dispatch_to_module(
    node: torch.fx.Node,
    builder: StructuredSDFGBuilder,
    metadata: TensorMetadata,
) -> None:
    """
    Dispatches to the GraphParser module for the parsing step. Throws if there is no module
    registered for the operation.
    """
    op: str = get_node_target_name(node.target)
    if not op in GRAPH_PARSER_MODULES:
        raise GraphParserErrorBase(
            node, f"Tried to dispatch module but it isn't registered: {op}"
        )
    GRAPH_PARSER_MODULES[op].parse(node, builder, metadata)
