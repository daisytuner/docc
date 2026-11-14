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

# Floating-point primitive types (including half precision).
FLOAT_PRIMITIVE_TYPES = frozenset(
    {
        PrimitiveType.Double,
        PrimitiveType.Float,
        PrimitiveType.Half,
        PrimitiveType.BFloat,
    }
)

# Integer primitive types (signed and unsigned).
INT_PRIMITIVE_TYPES = frozenset(
    {
        PrimitiveType.Int64,
        PrimitiveType.Int32,
        PrimitiveType.Int16,
        PrimitiveType.Int8,
        PrimitiveType.UInt64,
        PrimitiveType.UInt32,
        PrimitiveType.UInt16,
        PrimitiveType.UInt8,
    }
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

    scalar = scalar_type_for_dtype(python_type)
    if scalar is not None:
        return scalar

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


def _element_type_from_dtype_value(value):
    """Map a runtime dtype value to an SDFG scalar element type.

    Accepts a numpy scalar type (e.g. ``np.float64``), a python type
    (``float``/``int``/``bool``), or an ``np.dtype`` instance. Returns None if
    the value is not a recognizable dtype.
    """
    try:
        np_dtype = np.dtype(value)
    except (TypeError, ValueError):
        return None
    try:
        return sdfg_type_from_type(np_dtype.type)
    except ValueError:
        return None


def element_type_from_ast_node(ast_node, container_table=None, globals_dict=None):
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
        # Resolve a module-global dtype alias, e.g. `RealT = np.float64` used
        # as `np.zeros(shape, RealT)`. The global may hold a numpy scalar type,
        # a python type, or an np.dtype instance.
        if globals_dict is not None and ast_node.id in globals_dict:
            elem_type = _element_type_from_dtype_value(globals_dict[ast_node.id])
            if elem_type is not None:
                return elem_type

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
    left_is_float = left_pt in FLOAT_PRIMITIVE_TYPES
    right_is_float = right_pt in FLOAT_PRIMITIVE_TYPES

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


def _adapt_weak_to_concrete(weak_type, concrete_type):
    """Resolve a weak (Python-literal) operand against a concrete operand.

    Follows NEP 50: a Python scalar takes the *kind* of its literal but the
    *precision* of the concrete operand.
      - weak float  + concrete float  -> concrete float  (1.0 + f32 -> f32)
      - weak float  + concrete int    -> weak float's default precision
      - weak int    + concrete float  -> concrete float  (1   + f32 -> f32)
      - weak int    + concrete int    -> concrete int    (1   + i32 -> i32)
    """
    concrete_pt = concrete_type.primitive_type
    weak_is_float = weak_type.primitive_type in FLOAT_PRIMITIVE_TYPES
    concrete_is_float = concrete_pt in FLOAT_PRIMITIVE_TYPES

    if weak_is_float and not concrete_is_float:
        # A weak float combined with a concrete integer yields the default float.
        return weak_type
    # Otherwise the result adopts the concrete operand's precision/kind.
    return Scalar(concrete_pt)


def promote_scalar_types(left_type, left_weak, right_type, right_weak):
    """Promote two scalar operand types following NEP 50 weak-scalar rules.

    A "weak" operand is a Python literal (e.g. ``2.0`` or ``5``) whose type
    adapts to the concrete operand instead of forcing a wider result. This is
    what keeps ``float32_scalar * 2.0`` in float32 rather than upcasting to
    double.

    When both operands are weak (two literals) or both are concrete, this falls
    back to the standard :func:`promote_element_types` rules.
    """
    if left_weak and not right_weak:
        return _adapt_weak_to_concrete(left_type, right_type)
    if right_weak and not left_weak:
        return _adapt_weak_to_concrete(right_type, left_type)
    # Both weak or both concrete: standard promotion.
    return promote_element_types(left_type, right_type)


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


_DTYPE_TO_PRIMITIVE = None


def _dtype_to_primitive_table():
    """Single source of truth mapping numpy dtypes to SDFG primitive types."""
    global _DTYPE_TO_PRIMITIVE
    if _DTYPE_TO_PRIMITIVE is None:
        table = {
            np.dtype(np.float64): PrimitiveType.Double,
            np.dtype(np.float32): PrimitiveType.Float,
            np.dtype(np.float16): PrimitiveType.Half,
            np.dtype(np.bool_): PrimitiveType.Bool,
            np.dtype(np.int64): PrimitiveType.Int64,
            np.dtype(np.int32): PrimitiveType.Int32,
            np.dtype(np.int16): PrimitiveType.Int16,
            np.dtype(np.int8): PrimitiveType.Int8,
            np.dtype(np.uint64): PrimitiveType.UInt64,
            np.dtype(np.uint32): PrimitiveType.UInt32,
            np.dtype(np.uint16): PrimitiveType.UInt16,
            np.dtype(np.uint8): PrimitiveType.UInt8,
        }
        try:
            import ml_dtypes

            table[np.dtype(ml_dtypes.bfloat16)] = PrimitiveType.BFloat
        except Exception:
            pass
        _DTYPE_TO_PRIMITIVE = table
    return _DTYPE_TO_PRIMITIVE


def scalar_type_for_dtype(dtype_like):
    """Map a numpy dtype / scalar type / python numeric type to a Scalar.

    Returns None when the value is not a recognizable numeric dtype so callers
    can fall back to their own handling (e.g. ndarray annotations, structures).
    """
    try:
        np_dtype = np.dtype(dtype_like)
    except (TypeError, ValueError):
        return None
    primitive = _dtype_to_primitive_table().get(np_dtype)
    if primitive is None:
        return None
    return Scalar(primitive)


class TypeSystem:
    """Single authority for scalar type decisions in the Python frontend.

    Every visitor and handler resolves operand types, literal types and binary
    result types through one instance so the rules stay consistent. It holds a
    reference to the parser's ``container_table`` (and ``tensor_table``) which
    are mutated in place as the SDFG is built.
    """

    def __init__(
        self,
        container_table,
        tensor_table=None,
        default_float="float64",
        default_int="int64",
    ):
        self.container_table = container_table
        self.tensor_table = tensor_table if tensor_table is not None else {}
        self.default_float = sdfg_type_from_type(default_float)
        self.default_int = sdfg_type_from_type(default_int)
        if self.default_float is None or self.default_float.primitive_type not in (
            PrimitiveType.Float,
            PrimitiveType.Double,
        ):
            raise ValueError(f"Invalid default_float {default_float!r}")
        if self.default_int is None or self.default_int.primitive_type not in (
            PrimitiveType.Int32,
            PrimitiveType.Int64,
        ):
            raise ValueError(f"Invalid default_int {default_int!r}")

    @staticmethod
    def _base_name(operand):
        # Strip functional-index notation (e.g. `C(i, j)` -> `C`).
        if "(" in operand and operand.endswith(")"):
            return operand.split("(")[0]
        return operand

    def is_int(self, operand):
        try:
            if operand.lstrip("-").isdigit():
                return True
        except (AttributeError, ValueError):
            pass

        name = self._base_name(operand)
        if name in self.container_table:
            t = self.container_table[name]

            if isinstance(t, Scalar):
                return t.primitive_type in INT_PRIMITIVE_TYPES

            if type(t).__name__ == "Array" and hasattr(t, "element_type"):
                et = t.element_type
                if callable(et):
                    et = et()
                if isinstance(et, Scalar):
                    return et.primitive_type in INT_PRIMITIVE_TYPES

            if type(t).__name__ == "Pointer":
                for attr in ("pointee_type", "element_type"):
                    if hasattr(t, attr):
                        et = getattr(t, attr)
                        if callable(et):
                            et = et()
                        if isinstance(et, Scalar):
                            return et.primitive_type in INT_PRIMITIVE_TYPES

        return False

    def is_weak_literal(self, operand):
        # A "weak" operand is a bare Python literal (e.g. `2.0`, `5`, `true`)
        # that is not backed by a typed container. Following NEP 50 such
        # literals adopt the precision of the concrete operand they combine
        # with, so they never upcast a float32 value to double.
        return self._base_name(operand) not in self.container_table

    def element_type(self, name):
        lookup = self._base_name(name)
        if lookup in self.container_table:
            return element_type_from_sdfg_type(self.container_table[lookup])
        # Bare literal constant.
        return self.literal_string_type(name)

    def literal_string_type(self, literal, *, use_defaults=True):
        """Type of a bare literal; explicit casts read it at Python's precision."""
        if self.is_int(literal):
            return (
                self.constant_type(int(literal))
                if use_defaults
                else Scalar(PrimitiveType.Int64)
            )
        if literal in ("true", "false"):
            return Scalar(PrimitiveType.Bool)
        return self.default_float if use_defaults else Scalar(PrimitiveType.Double)

    def constant_type(self, value):
        """Element type of a Python constant value.

        ``bool`` is checked before ``int`` because ``bool`` is a subclass of
        ``int`` in Python.
        """
        if isinstance(value, np.generic):
            dtype = scalar_type_for_dtype(value.dtype)
            if dtype is not None:
                return dtype
        if isinstance(value, bool):
            return Scalar(PrimitiveType.Bool)
        if isinstance(value, int):
            limits = np.iinfo(
                np.int32
                if self.default_int.primitive_type == PrimitiveType.Int32
                else np.int64
            )
            if not limits.min <= value <= limits.max:
                raise OverflowError(
                    f"Integer literal {value} is outside the range of {limits.dtype}"
                )
            return self.default_int
        if isinstance(value, float):
            return self.default_float
        raise NotImplementedError(f"Cannot infer type for constant {value!r}")

    def result_type(self, left, right, op):
        """Result element type of a binary op, using NEP 50 weak-scalar rules."""
        left_is_int = self.is_int(left)
        right_is_int = self.is_int(right)
        if left_is_int and right_is_int:
            # true division / power of integers yields a float (NumPy semantics).
            if op in ("/", "**"):
                return self.default_float

        if (
            self.is_weak_literal(left)
            and self.is_weak_literal(right)
            and (
                self.element_type(left).primitive_type in FLOAT_PRIMITIVE_TYPES
                or self.element_type(right).primitive_type in FLOAT_PRIMITIVE_TYPES
            )
        ):
            return self.default_float

        dtype = promote_scalar_types(
            self.element_type(left),
            self.is_weak_literal(left),
            self.element_type(right),
            self.is_weak_literal(right),
        )
        # A float division/power of an otherwise-integer result is still float.
        if op in ("/", "**") and dtype.primitive_type in INT_PRIMITIVE_TYPES:
            return self.default_float
        return dtype

    def is_array(self, operand):
        return operand in self.tensor_table

    def promote(self, left, right):
        """Element-type promotion for array-array operations."""
        return promote_element_types(self.element_type(left), self.element_type(right))

    def promote_many(self, operands):
        """Left-to-right element-type promotion over several operands."""
        operands = list(operands)
        dtype = self.element_type(operands[0])
        for name in operands[1:]:
            dtype = promote_element_types(dtype, self.element_type(name))
        return dtype

    def array_result_type(self, left, right):
        """NumPy elementwise result type: a scalar adapts to an array operand."""
        return numpy_promote_types(
            self.element_type(left),
            self.is_array(left),
            self.element_type(right),
            self.is_array(right),
        )

    def scalar_type_for_dtype(self, dtype_like):
        return scalar_type_for_dtype(dtype_like)
