"""PyTorch GraphModule Parser utilities

This module contains the utilities and base classes for the PyTorch GraphModule Parser.
"""

import torch
import torch.fx
import torch.fx.passes.shape_prop
from torch.fx.node import Argument, Target

from typing import Any
from abc import ABC, abstractmethod

from docc.sdfg import (
    Type,
    PrimitiveType,
    Scalar,
    Pointer,
    Tensor,
    DebugInfo,
    StructuredSDFGBuilder,
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


class ContainerInfoBase:
    """
    This is the base class for container information. All the different real information classes
    inherit from this class. All available methods throw an exception.
    """

    def name(self) -> str:
        """Returns the name of the container"""
        raise NotImplementedError(
            "Cannot get sdfg_type from: " + self.__class__.__name__
        )

    def sdfg_type(self) -> Type:
        """
        Returns the SDFG type. This is the container type in the SDFG, e.g., Scalar, Pointer, etc.
        - not the Tensor type.
        """
        raise NotImplementedError(
            "Cannot get sdfg_type from: " + self.__class__.__name__
        )

    def sdfg_tensor_type(self) -> Tensor | None:
        """
        Returns the SDFG tensor type if available. If no tensor type is available None is returned.
        """
        raise NotImplementedError(
            "Cannot get sdfg_tensor_type from: " + self.__class__.__name__
        )

    def memory_managed(self) -> bool:
        """
        Returns True iff the container is memory manged, i.e., is was allocated during the parsing
        of the GraphModule.
        """
        raise NotImplementedError(
            "Cannot get memory_managed from: " + self.__class__.__name__
        )

    def in_argument(self) -> bool:
        """Returns True iff the container is an input argument"""
        raise NotImplementedError(
            "Cannot get in_argument from: " + self.__class__.__name__
        )

    def out_argument(self) -> bool:
        """Returns True iff the container is an output argument"""
        raise NotImplementedError(
            "Cannot get out_argument from: " + self.__class__.__name__
        )


class ContainerInfo(ContainerInfoBase):
    """
    This is the normal container information class. All fields from the base class can be accessed.
    """

    _name: str
    _sdfg_type: Type
    _sdfg_tensor_type: Tensor | None
    _memory_managed: bool
    _in_argument: bool
    _out_argument: bool

    def __init__(
        self,
        name: str,
        sdfg_type: Type,
        sdfg_tensor_type: Tensor | None,
        memory_managed: bool = False,
        in_argument: bool = False,
        out_argument: bool = False,
    ) -> None:
        """Initialization. By default all flags are False."""
        self._name: str = name
        self._sdfg_type: Type = sdfg_type
        self._sdfg_tensor_type: Tensor | None = sdfg_tensor_type
        self._memory_managed: bool = memory_managed
        self._in_argument: bool = in_argument
        self._out_argument: bool = out_argument

    def __str__(self) -> str:
        """Prints the container information as a string. Helpful for debugging purposes."""
        return (
            "ContainerInfo('"
            + str(self._name)
            + "', "
            + str(self._sdfg_type)
            + ", "
            + str(self._sdfg_tensor_type)
            + ", memory_managed="
            + str(self._memory_managed)
            + ", in_argument="
            + str(self._in_argument)
            + ", out_argument="
            + str(self._out_argument)
            + ")"
        )

    def name(self) -> str:
        """Returns the name of the container"""
        return self._name

    def sdfg_type(self) -> Type:
        """
        Returns the SDFG type. This is the container type in the SDFG, e.g., Scalar, Pointer, etc.
        - not the Tensor type.
        """
        return self._sdfg_type

    def sdfg_tensor_type(self) -> Tensor | None:
        """
        Returns the SDFG tensor type if available. If no tensor type is available None is returned.
        """
        return self._sdfg_tensor_type

    def memory_managed(self) -> bool:
        """
        Returns True iff the container is memory manged, i.e., is was allocated during the parsing
        of the GraphModule.
        """
        return self._memory_managed

    def in_argument(self) -> bool:
        """Returns True iff the container is an input argument"""
        return self._in_argument

    def out_argument(self) -> bool:
        """Returns True iff the container is an output argument"""
        return self._out_argument

    @staticmethod
    def from_tuple(
        name: str,
        sdfg_types: tuple[Type, Tensor | None],
        memory_managed: bool = False,
        in_argument: bool = False,
        out_argument: bool = False,
    ) -> "ContainerInfo":
        """
        Static helper method to create a container information directly from the SDFG type tuple.
        """
        return ContainerInfo(
            name,
            sdfg_types[0],
            sdfg_types[1],
            memory_managed=memory_managed,
            in_argument=in_argument,
            out_argument=out_argument,
        )

    def update(
        self,
        sdfg_type: Type | None = None,
        sdfg_tensor_type: Tensor | None = None,
        memory_managed: bool | None = None,
        in_argument: bool | None = None,
        out_argument: bool | None = None,
    ) -> "ContainerInfo":
        """
        Updates the container information. Each value is updated only if it is not None. Also
        returns the updated container information.
        """
        if not sdfg_type is None:
            self._sdfg_type: Type = sdfg_type
        if not sdfg_tensor_type is None:
            self._sdfg_tensor_type: Tensor | None = sdfg_tensor_type
        if not memory_managed is None:
            self._memory_managed: bool = memory_managed
        if not in_argument is None:
            self._in_argument: bool = in_argument
        if not out_argument is None:
            self._out_argument: bool = out_argument
        return self

    @staticmethod
    def copy(
        info: ContainerInfoBase,
        sdfg_type: Type | None = None,
        sdfg_tensor_type: Tensor | None = None,
        memory_managed: bool | None = None,
        in_argument: bool | None = None,
        out_argument: bool | None = None,
    ) -> "ContainerInfo":
        """
        Copies the container information. Each value from the original container information is
        copied if the matching argument of this function is None. Otherwise the copy contains the
        value of the argument. By default all values are copied from the orginial container
        information.
        """
        return ContainerInfo(
            info.name(),
            info.sdfg_type() if sdfg_type is None else sdfg_type,
            info.sdfg_tensor_type() if sdfg_tensor_type is None else sdfg_tensor_type,
            info.memory_managed() if memory_managed is None else memory_managed,
            info.in_argument() if in_argument is None else in_argument,
            info.out_argument() if out_argument is None else out_argument,
        )


class ContainerRefInfo(ContainerInfoBase):
    """
    Holds a reference to the actual container information. This is used if a container only exists
    virtually, i.e., was created from a tensor that shares its underlying data with another tensor.
    All access functions are mapped to the functions of the referenced container information.
    """

    _ref: ContainerInfo

    def __init__(self, ref: ContainerInfo) -> None:
        """Initialization"""
        self._ref: ContainerInfo = ref

    def __str__(self) -> str:
        """Prints the container information as a string. Helpful for debugging purposes."""
        return "ContainerRefInfo(" + str(self._ref) + ")"

    def ref(self) -> ContainerInfo:
        """Returns the container information"""
        return self._ref

    def name(self) -> str:
        """Returns the name of the container"""
        return self._ref.name()

    def sdfg_type(self) -> Type:
        """
        Returns the SDFG type. This is the container type in the SDFG, e.g., Scalar, Pointer, etc.
        - not the Tensor type.
        """
        return self._ref.sdfg_type()

    def sdfg_tensor_type(self) -> Tensor | None:
        """
        Returns the SDFG tensor type if available. If no tensor type is available None is returned.
        """
        return self._ref.sdfg_tensor_type()

    def memory_managed(self) -> bool:
        """
        Returns True iff the container is memory manged, i.e., is was allocated during the parsing
        of the GraphModule.
        """
        return self._ref.memory_managed()

    def in_argument(self) -> bool:
        """Returns True iff the container is an input argument"""
        return self._ref.in_argument()

    def out_argument(self) -> bool:
        """Returns True iff the container is an output argument"""
        return self._ref.out_argument()


class ContainerPreInfo(ContainerInfoBase):
    """
    Contains container information before the container is actually handled or created. At this
    point we do not care about the types but only about virtual containers in the form of references
    and if the container is an input or output container.
    """

    _name: str
    _ref: str | None
    _refed_by: str | None
    _in_argument: bool
    _out_argument: bool

    def __init__(
        self,
        name: str,
        ref: str | None = None,
        refed_by: str | None = None,
        in_argument: bool = False,
        out_argument: bool = False,
    ) -> None:
        """Initialization"""
        self._name: str = name
        self._ref: str | None = ref
        self._refed_by: str | None = refed_by
        self._in_argument: bool = in_argument
        self._out_argument: bool = out_argument

    def __str__(self) -> str:
        """Prints the container information as a string. Helpful for debugging purposes."""
        return (
            "ContainerPreInfo('"
            + str(self._name)
            + "', ref="
            + ("None" if self._ref is None else "'" + str(self._ref) + "'")
            + ", refed_by="
            + ("None" if self._refed_by is None else "'" + str(self._refed_by) + "'")
            + ", in_argument="
            + str(self._in_argument)
            + ", out_argument="
            + str(self._out_argument)
            + ")"
        )

    def name(self) -> str:
        """Returns the container information"""
        return self._name

    def is_ref(self) -> bool:
        """Returns True iff the container refrences another container"""
        return not self._ref is None

    def ref(self) -> str:
        """Returns the name of the referenced container. If there is non, an exception is thrown."""
        if self._ref is None:
            raise ValueError("ContainerPreInfo: Cannot get None-typed ref value")
        return self._ref

    def is_refed_by(self) -> bool:
        """Returns True iff the container is refrenced by another container"""
        return not self._refed_by is None

    def refed_by(self) -> str:
        """
        Returns the name of the container that references this container. If there is non, an
        exception is thrown.
        """
        if self._refed_by is None:
            raise ValueError("ContainerPreInfo: Cannot get None-typed refed_by value")
        return self._refed_by

    def in_argument(self) -> bool:
        """Returns True iff the container is an input argument"""
        return self._in_argument

    def out_argument(self) -> bool:
        """Returns True iff the container is an output argument"""
        return self._out_argument

    @staticmethod
    def copy(
        pre_info: "ContainerPreInfo",
        ref: str | None = None,
        refed_by: str | None = None,
        in_argument: bool | None = None,
        out_argument: bool | None = None,
    ) -> "ContainerPreInfo":
        """
        Copies the container information. Each value from the original container information is
        copied if the matching argument of this function is None. Otherwise the copy contains the
        value of the argument. By default all values are copied from the orginial container
        information.
        """
        return ContainerPreInfo(
            pre_info._name,
            pre_info._ref if ref is None else ref,
            pre_info._refed_by if refed_by is None else refed_by,
            pre_info._in_argument if in_argument is None else in_argument,
            pre_info._out_argument if out_argument is None else out_argument,
        )


class ContainerInfos:
    """
    A dictonary that maps containers (str) to container information (ContainerInfoBase). The
    information contain SDFG type, SDFG tensor type (if available), and flags about their lifetime.
    """

    _data: dict[str, ContainerInfoBase]

    def __init__(self) -> None:
        """Initialization"""
        self._data = {}

    def __getitem__(self, container: str) -> ContainerInfoBase:
        """Returns container information to the given container key"""
        return self._data[container]

    def __setitem__(self, container: str, info: ContainerInfoBase) -> None:
        """Sets container information to the given container key"""
        self._data[container] = info

    def __contains__(self, container: str) -> bool:
        """Returns True iff there is container information available to the given container key"""
        return container in self._data

    def __str__(self) -> str:
        """
        Prints the whole dictonary to a readable format. Helpful for debuging purposes.
        """
        result = "{"
        first: bool = True
        for container, info in self._data.items():
            if first:
                result += "\n"
                first: bool = False
            else:
                result += ",\n"
            result += "    '" + container + "': " + str(info)
        result += "\n}"
        return result

    def get_shape_str(self, container: str) -> str:
        """Get the shape string for a given container in the SDFG metadata format"""
        if not container in self._data:
            raise KeyError(f"No container '{container}' in container infos")
        sdfg_tensor_type: Tensor | None = self._data[container].sdfg_tensor_type()
        if sdfg_tensor_type is None:
            return ""
        else:
            return "[" + ",".join(sdfg_tensor_type.shape) + "]"

    def memory_managed(self) -> list[str]:
        """
        Returns a list of containers that are memory managed, i.e., that were allocated during the
        parsing of the GraphModule. This is helpful for handling deallocation of that memory at the
        end of the program.
        """
        return [
            container
            for container, info in self._data.items()
            if isinstance(info, ContainerInfo) and info.memory_managed()
        ]


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

    def determine_sdfg_type(
        self, node: torch.fx.Node, input: Any
    ) -> tuple[Type, Tensor | None]:
        """
        Tries to convert a PyTorch type to an SDFG type. The output is a pair of the SDFG type and
        an optional SDFG Tensor type if available. If the conversion fails, an exception is thrown.
        """
        if isinstance(input, torch.Tensor):
            base_type: Scalar = self.determine_sdfg_scalar_type(node, input.dtype)
            tensor_shape: list[str] = [str(dim) for dim in input.shape]
            tensor_stride: list[str] = [str(stride) for stride in input.stride()]
            if len(tensor_shape) == 0:
                sdfg_type: Type = base_type
            else:
                sdfg_type: Type = Pointer(base_type)
            return sdfg_type, Tensor(base_type, tensor_shape, tensor_stride)
        # Fallback to scalar types
        return self.determine_sdfg_scalar_type(node, input), None

    def convert_tensor_meta(
        self,
        node: torch.fx.Node,
        tensor_meta: torch.fx.passes.shape_prop.TensorMetadata,
    ) -> tuple[Type, Tensor]:
        """
        Converts a PyTorch TensorMetadata to a pair of the SDFG container type and the SDFG Tensor
        type.
        """
        base_type: Scalar = self.determine_sdfg_scalar_type(node, tensor_meta.dtype)
        tensor_shape: list[str] = [str(dim) for dim in tensor_meta.shape]
        tensor_stride: list[str] = [str(stride) for stride in tensor_meta.stride]
        if len(tensor_shape) == 0:
            sdfg_type: Type = base_type
        else:
            sdfg_type: Type = Pointer(base_type)
        return sdfg_type, Tensor(base_type, tensor_shape, tensor_stride)

    def convert_arg_to_container(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        arg: Argument,
        resolve: bool = True,
    ) -> str:
        """
        Tries to convert a PyTorch Argument to an SDFG container (str). If it fails, an exception is
        thrown. By default this also resolves the container name with the container information,
        i.e., if the container is a virtual container that references another container, this
        reference is resolved. This can be prevented by setting the resolve flag to False.
        """
        container: str | None = None
        if isinstance(arg, torch.fx.Node):
            container: str | None = str(arg)
        if container is None:
            raise GraphParserError(
                self, node, f"Cannot convert argument to container: {type(arg)}"
            )
        if resolve and container in container_info:
            return container_info[container].name()
        else:
            return container

    def convert_arg_to_constant(
        self, node: torch.fx.Node, arg: Argument
    ) -> tuple[str, Scalar]:
        """
        Tries to convert a PyTorch Argument to an SDFG constant. The constant is a pair of the
        constant's value (str) and its Scalar type. If it fails, an exception is thrown.
        """
        if isinstance(arg, (int, float, bool, torch.dtype)):
            return str(arg), self.determine_sdfg_scalar_type(node, arg)
        raise GraphParserError(
            self, node, f"Cannot convert argument to constant: {type(arg)}"
        )

    def convert_arg_to_sdfg_value(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        arg: Argument,
        resolve: bool = True,
    ) -> str | tuple[str, Scalar]:
        """
        Tries to convert a PyTorch Argument to an SDFG value, i.e., either a container or an SDFG
        constant. If the PyTorch Argument is a SDFG container, the container name (str) is returned.
        If the PyTorch Argument is a SDFG constant, the constant's value and its Scalar type are
        returned. If it fails, an exception is thrown.
        """
        if isinstance(arg, torch.fx.Node):
            return self.convert_arg_to_container(
                node, container_info, arg, resolve=resolve
            )
        elif isinstance(arg, (int, float, bool, torch.dtype)):
            return self.convert_arg_to_constant(node, arg)
        raise GraphParserError(
            self, node, f"Cannot convert argument to SDFG value: {type(arg)}"
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

    def parse_torch_stack_trace(self, stack_trace: str) -> DebugInfo:
        """
        Parses a PyTorch stack trace to SDFG debug information. If the parsing fails, an empty
        debug information is returned.
        """
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


class GraphParserModule(GraphParserBase, ABC):
    """
    This is the base class for a module in the PyTorch GraphModule Parser. For each operation a
    GraphParser module can be registered. The job of the module is to parse only its registered
    operations. This base class provides all the helper and utility function needed for that.
    """

    def pre_parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        """
        This function is called from the GraphParser to dispatch to the GraphParser module. Its
        purpose is to gather information about the virtual tensors referencing each other and store
        it into the container information.
        """
        pass

    @abstractmethod
    def parse(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> None:
        """
        This function is called from the GraphParser to dispatch to the GraphParser module. Its
        purpose is to translate the provided operation into an equivalent SDFG operation.
        """
        pass

    def get_debug_info(self, node: torch.fx.Node) -> DebugInfo:
        """
        Converts the PyTorch stack trace attached to the node to SDFG debug information if available
        """
        return self.parse_torch_stack_trace(
            "" if not "stack_trace" in node.meta else node.meta["stack_trace"]
        )

    def get_arg_container(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        index: int,
        resolve: bool = True,
    ) -> str:
        """
        Convert the index-th PyTorch Argument to an SDFG container. Throws an exception if the index
        is out of bounds. See ``convert_arg_to_container`` for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        return self.convert_arg_to_container(
            node, container_info, node.args[index], resolve=resolve
        )

    def get_arg_sdfg_value(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        index: int,
        resolve: bool = True,
    ) -> str | tuple[str, Scalar]:
        """
        Convert the index-th PyTorch Argument to an SDFG value. Throws an exception if the index is
        out of bounds. See ``convert_arg_to_sdfg_value`` for more information.
        """
        if index >= len(node.args):
            raise GraphParserError(
                self,
                node,
                f"Tried to get the {index+1}. argument but has only {len(node.args)}",
            )
        return self.convert_arg_to_sdfg_value(
            node, container_info, node.args[index], resolve=resolve
        )

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

    def get_scalar_type(
        self, node: torch.fx.Node, container_info: ContainerInfos, container: str
    ) -> Scalar:
        """
        Returns the Scalar type of a container. If the container does not have a Scalar type, an
        exception is thrown.
        """
        if not container in container_info:
            raise GraphParserError(
                self,
                node,
                f"Cannot get container info for container '{container}'",
            )
        sdfg_type: Type = container_info[container].sdfg_type()
        if not isinstance(sdfg_type, Scalar):
            raise GraphParserError(
                self,
                node,
                f"No scalar type available for container '{container}'",
            )
        return sdfg_type

    def get_tensor_type(
        self, node: torch.fx.Node, container_info: ContainerInfos, container: str
    ) -> Tensor:
        """
        Returns the Tensor type of a container. If the container does not have a Tensor type, an
        exception is thrown.
        """
        if not container in container_info:
            raise GraphParserError(
                self,
                node,
                f"Cannot get container info for container '{container}'",
            )
        tensor_type: Tensor | None = container_info[container].sdfg_tensor_type()
        if tensor_type is None:
            raise GraphParserError(
                self,
                node,
                f"No tensor type available for container '{container}",
            )
        return tensor_type

    def allocate_memory(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
        container: str,
        debug_info: DebugInfo | None = None,
    ) -> None:
        """
        Adds a memory allocation (malloc) to the SDFG for the given container. The size is obtained
        from the Tensor type. If the size is 0 (Scalar Tensor), this function is a NOP.
        """
        if debug_info is None:
            debug_info_: DebugInfo = self.get_debug_info(node)
        else:
            debug_info_: DebugInfo = debug_info
        if not container in container_info:
            raise GraphParserError(
                self,
                node,
                "Could not allocate memory because container does not exist: "
                + container,
            )
        info: ContainerInfoBase = container_info[container]
        if not isinstance(info, ContainerInfo):
            raise GraphParserError(
                self,
                node,
                "Expected ContainerInfo but got: " + str(type(info)),
            )
        sdfg_tensor_type: Tensor | None = info.sdfg_tensor_type()
        if sdfg_tensor_type is None:
            raise GraphParserError(
                self,
                node,
                "Could not allocate memory for non-tensor container: " + container,
            )
        size: str = sdfg_tensor_type.total_size()
        if size != "0":
            builder.add_malloc_block(container, size, debug_info_)
            info.update(memory_managed=True)

    def resolve_contaner_name_forward(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        container: str,
        sdfg_types: tuple[Type, Tensor | None],
    ) -> ContainerInfo:
        """
        Uses the container information to forward resolve a container name. Therefore, the container
        pre-information is evaluated by following the "reference" field until an already created
        container is reached. All traversed (so called virtual) containers are marked as reference
        to the found container. Returns the container information of the found container.
        """
        current: str = container
        ref_containers: list[str] = []
        out_argument: bool = False
        while current in container_info:
            info: ContainerInfoBase = container_info[current]
            out_argument: bool = out_argument or info.out_argument()
            if isinstance(info, ContainerPreInfo):
                if info.is_ref():
                    ref_containers.append(current)
                    current: str = info.ref()
                else:
                    break
            else:
                current = info.name()
                break
        if current in container_info:
            info: ContainerInfoBase = container_info[current]
            if isinstance(info, ContainerPreInfo):
                current_info: ContainerInfo = ContainerInfo.from_tuple(
                    current, sdfg_types, out_argument=out_argument
                )
            elif isinstance(info, ContainerInfo):
                current_info: ContainerInfo = info.update(out_argument=out_argument)
            elif isinstance(info, ContainerRefInfo):
                current_info: ContainerInfo = info.ref().update(
                    out_argument=out_argument
                )
            else:
                raise GraphParserError(
                    self, node, "Cannot handle ContainerInfoBase: " + str(type(info))
                )
        else:
            current_info: ContainerInfo = ContainerInfo.from_tuple(
                current, sdfg_types, out_argument=out_argument
            )
        container_info[current] = current_info
        for ref_container in ref_containers:
            if ref_container in container_info:
                info: ContainerInfoBase = container_info[ref_container]
                if isinstance(info, ContainerPreInfo):
                    container_info[ref_container] = ContainerRefInfo(current_info)
                else:
                    raise GraphParserError(
                        self,
                        node,
                        "Expected ContainerPreInfo for ref container but got: "
                        + str(type(info)),
                    )
            else:
                container_info[ref_container] = ContainerRefInfo(current_info)
        return current_info

    def update_container_types(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
        container: str,
    ) -> None:
        """
        Updates the container types of a container in the container info. In theory, this does
        nothing to the generated SDFG but it only updates the container information. For example, if
        a tensor is transposed, the underlying data stays the same and only the shape and stride
        information must be updated to respect the transposition in the next operation. In practice,
        however, this can lead to the edge case that a container is now an input and output argument
        simultaneously. As this is not possible, a SDFG tensor copy node is generated in this case
        to explicitly copy the data.
        """
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        if "tensor_meta" in node.meta:
            sdfg_types: tuple[Type, Tensor | None] = self.convert_tensor_meta(
                node, node.meta["tensor_meta"]
            )
        else:
            sdfg_types: tuple[Type, Tensor | None] = self.determine_sdfg_type(
                node, node.meta["val"]
            )
        info: ContainerInfo = self.resolve_contaner_name_forward(
            node, container_info, container, sdfg_types
        )
        if info.in_argument() and info.out_argument():
            ref_info: ContainerInfoBase = container_info[container]
            if not isinstance(ref_info, ContainerRefInfo):
                raise GraphParserError(
                    self,
                    node,
                    "Expected ContainerRefInfo but got: " + str(type(ref_info)),
                )
            if ref_info.ref() != info:
                raise GraphParserError(
                    self,
                    node,
                    f"Container '{container}' is not a reference to container '{info.name()}'",
                )
            container_tensor: Tensor | None = sdfg_types[1]
            if container_tensor is None:
                raise GraphParserError(
                    self,
                    node,
                    "Cannot copy into non-tensor type for container: " + container,
                )
            sdfg_types: tuple[Type, Tensor | None] = (
                sdfg_types[0],
                Tensor(container_tensor.element_type, container_tensor.shape),
            )
            container_info[container] = ContainerInfo.from_tuple(
                container, sdfg_types, out_argument=True
            )
            info.update(out_argument=False)
            builder.add_container(container, sdfg_types[0], is_argument=True)
            debug_info: DebugInfo = self.get_debug_info(node)
            sdfg_tensor: Tensor | None = info.sdfg_tensor_type()
            if sdfg_tensor is None:
                raise GraphParserError(
                    self,
                    node,
                    "Cannot copy non-tensor type for container: " + info.name(),
                )
            print((info.name(), sdfg_tensor, container, sdfg_types[1]))
            builder.add_copy_op(
                info.name(), sdfg_tensor, container, sdfg_types[1], debug_info
            )
        else:
            info.update(sdfg_type=sdfg_types[0], sdfg_tensor_type=sdfg_types[1])

    def resolve_container_name_backward(
        self,
        node: torch.fx.Node,
        container_info: ContainerInfos,
        container: str,
        sdfg_types: tuple[Type, Tensor | None],
    ) -> ContainerInfo:
        """
        Uses the container information to backward resolve a container name. Therefore, the
        container pre-information is evaluated by following the "referenced by" field until an
        already created container is reached. All traversed (so called virtual) containers are
        marked as reference to the found container. Returns the container information of the found
        container.
        """
        current: str = container
        ref_containers: list[str] = []
        out_argument: bool = False
        while current in container_info:
            info: ContainerInfoBase = container_info[current]
            out_argument: bool = out_argument or info.out_argument()
            if isinstance(info, ContainerPreInfo):
                if info.is_refed_by():
                    ref_containers.append(current)
                    current: str = info.refed_by()
                else:
                    break
            else:
                current = info.name()
                break
        if current in container_info:
            info: ContainerInfoBase = container_info[current]
            if isinstance(info, ContainerPreInfo):
                current_info: ContainerInfo = ContainerInfo.from_tuple(
                    current, sdfg_types, out_argument=out_argument
                )
                container_info[current] = current_info
            elif isinstance(info, ContainerInfo):
                current_info: ContainerInfo = info.update(out_argument=out_argument)
            elif isinstance(info, ContainerRefInfo):
                current_info: ContainerInfo = info.ref().update(
                    out_argument=out_argument
                )
            else:
                raise GraphParserError(
                    self, node, "Cannot handle ContainerInfoBase: " + str(type(info))
                )
        else:
            current_info: ContainerInfo = ContainerInfo.from_tuple(
                current, sdfg_types, out_argument=out_argument
            )
            container_info[current] = current_info
        for ref_container in ref_containers:
            if ref_container in container_info:
                info: ContainerInfoBase = container_info[ref_container]
                if isinstance(info, ContainerPreInfo):
                    container_info[ref_container] = ContainerRefInfo(current_info)
                else:
                    raise GraphParserError(
                        self,
                        node,
                        "Expected ContainerPreInfo for ref container but got: "
                        + str(type(info)),
                    )
            else:
                container_info[ref_container] = ContainerRefInfo(current_info)
        return current_info

    def get_result_container(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> str:
        """
        Creates a new container to use as the result of the current operation. For creating a
        container for an intermediate result, see ``create_intermediate_container``. If the
        operation has multiple results (tuple), use ``create_result_containers``.
        """
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        if "tensor_meta" in node.meta:
            sdfg_types: tuple[Type, Tensor | None] = self.convert_tensor_meta(
                node, node.meta["tensor_meta"]
            )
        else:
            sdfg_types: tuple[Type, Tensor | None] = self.determine_sdfg_type(
                node, node.meta["val"]
            )
        info: ContainerInfo = self.resolve_container_name_backward(
            node, container_info, node.name, sdfg_types
        )
        if info.out_argument() and isinstance(sdfg_types[0], Scalar):
            info.update(sdfg_type=Pointer(sdfg_types[0]))
        builder.add_container(
            info.name(), info.sdfg_type(), is_argument=info.out_argument()
        )
        if not info.out_argument():
            self.allocate_memory(node, builder, container_info, info.name())
        return info.name()

    def get_result_containers(
        self,
        num: int,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
    ) -> tuple[str, ...]:
        """
        Creates new containers to use as the results of the current operation. Because SDFGs do not
        support tuples, a container for each element of the tuple is created. Access to them are
        resolved with the help of virtual containers. For creating only a single result of the
        current operation, see ``get_result_container``.
        """
        base_name: str = node.name
        containers: list[str] = []
        if not "val" in node.meta:
            raise GraphParserError(
                self,
                node,
                "No result type information in metadata",
            )
        if (not isinstance(node.meta["val"], tuple)) or len(node.meta["val"]) != num:
            raise GraphParserError(
                self, node, f"Not exactly {num} result type information in metadata"
            )
        for i in range(num):
            sdfg_types: tuple[Type, Tensor | None] = self.determine_sdfg_type(
                node, node.meta["val"][i]
            )
            info: ContainerInfo = self.resolve_container_name_backward(
                node, container_info, f"{base_name}_{i}", sdfg_types
            )
            builder.add_container(
                info.name(), sdfg_types[0], is_argument=info.out_argument()
            )
            if not info.out_argument():
                self.allocate_memory(node, builder, container_info, info.name())
            containers.append(info.name())
        return tuple(containers)

    def create_intermediate_container(
        self,
        node: torch.fx.Node,
        builder: StructuredSDFGBuilder,
        container_info: ContainerInfos,
        sdfg_type: Type,
        sdfg_tensor_type: Tensor | None = None,
    ) -> str:
        """
        Creates a container to use for intermediate results. Memory allocation and management is
        automatically handled. DO NOT use this method for creating result container(s). See
        ``get_result_container`` and ``get_result_containers`` for that.
        """
        container: str = builder.find_new_name("intermediate")
        container_info[container] = ContainerInfo(
            container, sdfg_type, sdfg_tensor_type
        )
        builder.add_container(container, sdfg_type)
        if not isinstance(sdfg_type, Scalar):
            self.allocate_memory(node, builder, container_info, container)
        return container

    def align_constant_type(
        self, node: torch.fx.Node, constant: tuple[str, Scalar], dst_type: Scalar
    ) -> Scalar:
        """
        Align an SDFG constant (pair of constant value and Scalar type) to a Scalar destination
        type. This is helpful for saving unnecessary casts. For example, if a tensor with base type
        float should be scaled by a double or integer constant elementwisely, this method would
        convert it to float.
        """
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
            if primitive_type_is_integer(constant[1].primitive_type):
                int_val: int = int(constant[0])
                if int_val >= limit[0] and int_val <= limit[1]:
                    return dst_type
        elif primitive_type_is_floating_point(dst_type.primitive_type):
            if primitive_type_is_integer(
                constant[1].primitive_type
            ) or primitive_type_is_floating_point(constant[1].primitive_type):
                return dst_type
        raise GraphParserError(
            self,
            node,
            f"Cannot align constant type: {constant[1].primitive_type} -> {dst_type.primitive_type}",
        )


GRAPH_PARSER_PRE_MODULES: dict[str, GraphParserModule] = {}
GRAPH_PARSER_MODULES: dict[str, GraphParserModule] = {}


def get_node_target_name(target: Target) -> str:
    """Helper method to convert a PyTorch Target to a string"""
    if isinstance(target, str):
        return target
    elif target.__module__ == "torch._ops.aten":
        return str(target)
    else:
        return target.__module__ + "." + target.__name__


def _register_module(
    parser_modules: dict[str, GraphParserModule], op: str, module: GraphParserModule
) -> None:
    """Register a module into the given dictonary. Throws if the module already exists."""
    if op in parser_modules:
        raise KeyError(
            f"GraphParser: Could not register module because it already exists: {op}"
        )
    parser_modules[op] = module


def register_pre_module(op: str, module: GraphParserModule) -> None:
    """
    Registers a GraphParser module to an operation for the pre-parsing step. Throws if another
    module is already registered to that operation.
    """
    _register_module(GRAPH_PARSER_PRE_MODULES, op, module)


def register_module(op: str, module: GraphParserModule) -> None:
    """
    Registers a GraphParser module to an operation for the parsing step. Throws if another module is
    already registered to that operation.
    """
    _register_module(GRAPH_PARSER_MODULES, op, module)


def dispatch_to_pre_module(
    node: torch.fx.Node,
    builder: StructuredSDFGBuilder,
    container_info: ContainerInfos,
) -> None:
    """
    Dispatches to the GraphParser module for the pre-parsing step. NOP if there is no module
    registered for the operation.
    """
    op: str = get_node_target_name(node.target)
    if op in GRAPH_PARSER_PRE_MODULES:
        GRAPH_PARSER_PRE_MODULES[op].pre_parse(node, builder, container_info)


def dispatch_to_module(
    node: torch.fx.Node,
    builder: StructuredSDFGBuilder,
    container_info: ContainerInfos,
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
    GRAPH_PARSER_MODULES[op].parse(node, builder, container_info)
