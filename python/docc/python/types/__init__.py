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
    """
    Promote two dtypes following NumPy rules for array-array operations.

    Rules:
    - float + float → wider float
    - int + int → wider int
    - float + int → float that can represent both (float32+int32 → float64)
    """
    left_pt = left_element_type.primitive_type
    right_pt = right_element_type.primitive_type

    # Check if types are floating point (includes half-precision types)
    float_types = {
        PrimitiveType.Double,
        PrimitiveType.Float,
        PrimitiveType.Half,
        PrimitiveType.BFloat,
    }
    int_types = {
        PrimitiveType.Int64,
        PrimitiveType.Int32,
        PrimitiveType.Int16,
        PrimitiveType.Int8,
        PrimitiveType.UInt64,
        PrimitiveType.UInt32,
        PrimitiveType.UInt16,
        PrimitiveType.UInt8,
    }

    left_is_float = left_pt in float_types
    right_is_float = right_pt in float_types

    # Both floats: return wider
    if left_is_float and right_is_float:
        if left_pt == PrimitiveType.Double or right_pt == PrimitiveType.Double:
            return Scalar(PrimitiveType.Double)
        if left_pt == PrimitiveType.Float or right_pt == PrimitiveType.Float:
            return Scalar(PrimitiveType.Float)
        # Half-precision types: same type stays, mixed promotes to Float
        if left_pt == right_pt:
            return Scalar(left_pt)  # BFloat+BFloat→BFloat, Half+Half→Half
        return Scalar(PrimitiveType.Float)  # Mixed half types → float32

    # Both integers: return wider (simplified - always Int64 for now)
    if not left_is_float and not right_is_float:
        if left_pt == PrimitiveType.Int64 or right_pt == PrimitiveType.Int64:
            return Scalar(PrimitiveType.Int64)
        if left_pt == PrimitiveType.UInt64 or right_pt == PrimitiveType.UInt64:
            return Scalar(PrimitiveType.Int64)  # Promote to signed for safety
        if left_pt == PrimitiveType.Int32 or right_pt == PrimitiveType.Int32:
            return Scalar(PrimitiveType.Int32)
        return Scalar(PrimitiveType.Int64)  # Default

    # Mixed float + int: need a float that can represent the int
    # float32 can represent int16/int8, but not int32
    # float64 can represent int32 and smaller
    # half types + int → promote to float32 (half can't represent ints well)
    float_type = left_pt if left_is_float else right_pt
    int_type = right_pt if left_is_float else left_pt

    # If float is already double, use double
    if float_type == PrimitiveType.Double:
        return Scalar(PrimitiveType.Double)

    # Half-precision + any int → float32 (half types can't represent ints well)
    if float_type in {PrimitiveType.Half, PrimitiveType.BFloat}:
        return Scalar(PrimitiveType.Float)

    # float32 + (int32 or larger) → float64
    if int_type in {
        PrimitiveType.Int32,
        PrimitiveType.Int64,
        PrimitiveType.UInt32,
        PrimitiveType.UInt64,
    }:
        return Scalar(PrimitiveType.Double)

    # float32 + (int16 or smaller) → float32
    return Scalar(PrimitiveType.Float)


def numpy_promote_types(left_type, left_is_array, right_type, right_is_array):
    """
    Implement NumPy's type promotion rules for binary operations.

    Key rule: Scalars adapt to arrays, not vice versa.
    - array + scalar → array's dtype (scalar is cast to array's dtype)
    - array + array → standard promotion (wider/float wins)
    - scalar + scalar → standard promotion

    Args:
        left_type: Element type of left operand (Scalar)
        left_is_array: True if left operand is an array
        right_type: Element type of right operand (Scalar)
        right_is_array: True if right operand is an array

    Returns:
        Result element type (Scalar)
    """
    if left_is_array and not right_is_array:
        # Scalar adapts to array
        return left_type
    if right_is_array and not left_is_array:
        # Scalar adapts to array
        return right_type
    # Both arrays or both scalars: use standard promotion
    return promote_element_types(left_type, right_type)
