import ast
import inspect
import numpy as np

from typing import get_origin, get_args
from docc.sdfg import (
    PrimitiveType,
    Scalar,
    Pointer,
    Array,
    Structure,
    Tensor,
    Type,
)


def sdfg_type_from_type(python_type):
    if isinstance(python_type, Type):
        return python_type

    # Handle numpy.ndarray[Shape, python_type] type annotations
    if get_origin(python_type) is np.ndarray:
        args = get_args(python_type)
        if len(args) >= 2:
            elem_type = sdfg_type_from_type(args[1])
            return Pointer(elem_type)
        # Unparameterized ndarray defaults to void pointer
        return Pointer(Scalar(PrimitiveType.Void))

    # Handle np.dtype[ScalarType] annotations
    if get_origin(python_type) is np.dtype:
        return sdfg_type_from_type(get_args(python_type)[0])

    if python_type is float or python_type is np.float64:
        return Scalar(PrimitiveType.Double)
    elif python_type is np.float32:
        return Scalar(PrimitiveType.Float)
    elif python_type is bool or python_type is np.bool_:
        return Scalar(PrimitiveType.Bool)
    elif python_type is int or python_type is np.int64:
        return Scalar(PrimitiveType.Int64)
    elif python_type is np.int32:
        return Scalar(PrimitiveType.Int32)
    elif python_type is np.int16:
        return Scalar(PrimitiveType.Int16)
    elif python_type is np.int8:
        return Scalar(PrimitiveType.Int8)
    elif python_type is np.uint64:
        return Scalar(PrimitiveType.UInt64)
    elif python_type is np.uint32:
        return Scalar(PrimitiveType.UInt32)
    elif python_type is np.uint16:
        return Scalar(PrimitiveType.UInt16)
    elif python_type is np.uint8:
        return Scalar(PrimitiveType.UInt8)

    # Handle Python classes - map to Structure type
    if inspect.isclass(python_type):
        return Pointer(Structure(python_type.__name__))

    raise ValueError(f"Cannot map type to SDFG type: {python_type}")


def element_type_from_sdfg_type(sdfg_type: Type):
    if isinstance(sdfg_type, Scalar):
        return sdfg_type
    elif isinstance(sdfg_type, (Pointer, Array, Tensor)):
        return Scalar(sdfg_type.primitive_type)
    else:
        raise ValueError(
            f"Unsupported SDFG type for element type extraction: {sdfg_type}"
        )


def element_type_from_ast_node(ast_node, container_table=None):
    # Default to double
    if ast_node is None:
        return Scalar(PrimitiveType.Double)

    # Handle python built-in types
    if isinstance(ast_node, ast.Name):
        if ast_node.id == "float":
            return Scalar(PrimitiveType.Double)
        if ast_node.id == "int":
            return Scalar(PrimitiveType.Int64)
        if ast_node.id == "bool":
            return Scalar(PrimitiveType.Bool)

    # Handle complex types
    if isinstance(ast_node, ast.Attribute):
        # Handle numpy types like np.float64, np.int32, etc.
        if isinstance(ast_node.value, ast.Name) and ast_node.value.id in [
            "numpy",
            "np",
        ]:
            if ast_node.attr == "float64":
                return Scalar(PrimitiveType.Double)
            if ast_node.attr == "float32":
                return Scalar(PrimitiveType.Float)
            if ast_node.attr == "int64":
                return Scalar(PrimitiveType.Int64)
            if ast_node.attr == "int32":
                return Scalar(PrimitiveType.Int32)
            if ast_node.attr == "int16":
                return Scalar(PrimitiveType.Int16)
            if ast_node.attr == "int8":
                return Scalar(PrimitiveType.Int8)
            if ast_node.attr == "uint64":
                return Scalar(PrimitiveType.UInt64)
            if ast_node.attr == "uint32":
                return Scalar(PrimitiveType.UInt32)
            if ast_node.attr == "uint16":
                return Scalar(PrimitiveType.UInt16)
            if ast_node.attr == "uint8":
                return Scalar(PrimitiveType.UInt8)
            if ast_node.attr == "bool_":
                return Scalar(PrimitiveType.Bool)

        # Handle arr.dtype - get element type from array's type in symbol table
        if ast_node.attr == "dtype" and container_table is not None:
            if isinstance(ast_node.value, ast.Name):
                var_name = ast_node.value.id
                if var_name in container_table:
                    var_type = container_table[var_name]
                    return element_type_from_sdfg_type(var_type)

    raise ValueError(f"Cannot map AST node to SDFG type: {ast.dump(ast_node)}")


def promote_element_types(left_element_type, right_element_type):
    """Promote two dtypes following NumPy rules: float > int, wider > narrower."""
    priority = {
        PrimitiveType.Double: 4,
        PrimitiveType.Float: 3,
        PrimitiveType.Int64: 2,
        PrimitiveType.Int32: 1,
    }
    left_prio = priority.get(left_element_type.primitive_type, 0)
    right_prio = priority.get(right_element_type.primitive_type, 0)
    if left_prio >= right_prio:
        return left_element_type
    else:
        return right_element_type
