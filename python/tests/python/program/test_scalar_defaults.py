import math
import numpy as np
import pytest

from docc.compiler.docc_program import DoccOptions
from docc.python import native
from docc.sdfg import PrimitiveType


def _literal_expressions(a):
    s = 0.0
    n = 3
    negative = -1.5
    m = -3
    mixed = 1 + 2.0
    quotient = 3 / 2
    negated = -n
    remainder = m % 2
    root = math.sqrt(4.0)
    for i in range(a.shape[0]):
        s = s + a[i]
    return s, n, negative, m, mixed, quotient, negated, remainder, root


@pytest.mark.parametrize(
    "options,float_type,int_type,expected_sum",
    [
        ({}, PrimitiveType.Double, PrimitiveType.Int64, 16777217.0),
        (
            {"default_float": "float32", "default_int": "int32"},
            PrimitiveType.Float,
            PrimitiveType.Int32,
            16777216.0,
        ),
    ],
)
def test_literal_defaults_and_arithmetic(options, float_type, int_type, expected_sum):
    program = native(_literal_expressions, **options)
    a = np.array([16777216.0, 1.0], dtype=np.float32)
    assert program(a) == (expected_sum, 3, -1.5, -3, 3.0, 1.5, -3, 1, 2.0)
    expected_types = [
        float_type,
        int_type,
        float_type,
        int_type,
        float_type,
        float_type,
        int_type,
        int_type,
        float_type,
    ]
    for i, dtype in enumerate(expected_types):
        assert (
            program.last_sdfg.type(f"_docc_ret_{i}").pointee_type.primitive_type
            == dtype
        )


def test_explicit_types_and_weak_promotion_are_preserved():
    @native(default_float="float32", default_int="int32")
    def kernel(a, b, scalar):
        explicit_float = np.float64(16777217.0)
        explicit_int = np.int64(2147483648)
        python_float = float(16777217.0)
        python_int = int(2147483648)
        default_array = np.zeros(2)
        weak = a + 1.0
        wide_weak = scalar + 16777217.0
        wide_array = b + 16777217.0
        return (
            explicit_float,
            explicit_int,
            python_float,
            python_int,
            default_array,
            weak,
            scalar,
            wide_weak,
            wide_array,
        )

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float64)
    results = kernel(a, b, np.float64(3.0))
    assert results[:4] == (16777217.0, 2147483648, 16777217.0, 2147483648)
    assert results[4].dtype == np.float64
    assert results[5].dtype == np.float32
    np.testing.assert_array_equal(results[5], a + 1.0)
    assert results[6] == 3.0
    assert results[7] == 16777220.0
    np.testing.assert_array_equal(results[8], b + 16777217.0)
    for i, dtype in [
        (0, PrimitiveType.Double),
        (1, PrimitiveType.Int64),
        (2, PrimitiveType.Double),
        (3, PrimitiveType.Int64),
        (6, PrimitiveType.Double),
    ]:
        assert (
            kernel.last_sdfg.type(f"_docc_ret_{i}").pointee_type.primitive_type == dtype
        )


def _inline_sum(a):
    s = 0.0
    for i in range(a.shape[0]):
        s = s + a[i]
    return s


def test_inlined_functions_inherit_defaults():
    @native(default_float="float32")
    def kernel(a):
        return _inline_sum(a)

    assert kernel(np.array([16777216.0, 1.0], dtype=np.float32)) == 16777216.0


@pytest.mark.parametrize("symbolic_shapes", [True, False])
def test_index_types_do_not_follow_literal_defaults(symbolic_shapes):
    @native(default_int="int32", symbolic_shapes=symbolic_shapes)
    def kernel(a):
        n = a.shape[0] - 1
        for i in range(n):
            a[i] = a[i] + 1

    sdfg = kernel.to_sdfg(np.zeros(4, dtype=np.float32))
    assert sdfg.type("n").primitive_type == PrimitiveType.Int64
    assert sdfg.type("i").primitive_type == PrimitiveType.Int64
    if symbolic_shapes:
        assert sdfg.type("_s0").primitive_type == PrimitiveType.Int64


def test_scalar_defaults_separate_memory_and_disk_caches():
    @native
    def kernel():
        return 16777217.0, 3

    wide = kernel.compile()
    assert wide() == (16777217.0, 3)
    kernel.options.default_float = "float32"
    narrow_float = kernel.compile()
    assert narrow_float() == (16777216.0, 3)
    kernel.options.default_int = "int32"
    narrow_int = kernel.compile()
    assert narrow_int() == (16777216.0, 3)
    assert len({id(wide), id(narrow_float), id(narrow_int)}) == 3
    assert len({wide.lib_path, narrow_float.lib_path, narrow_int.lib_path}) == 3


@pytest.mark.parametrize(
    "option,value",
    [
        ("default_float", "int32"),
        ("default_float", "float16"),
        ("default_int", "float32"),
        ("default_int", "uint32"),
    ],
)
def test_invalid_defaults_are_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        DoccOptions(**{option: value})


def test_narrow_integer_literal_overflow_is_rejected():
    @native(default_int="int32")
    def kernel():
        return 2147483648

    with pytest.raises(OverflowError, match="Integer literal.*int32"):
        kernel.compile()


def test_explicit_defaults_override_target_defaults(monkeypatch):
    monkeypatch.setattr(
        "docc.compiler.docc_program.target_default_scalar_types",
        lambda target: ("float32", "int32"),
    )
    options = DoccOptions()
    assert (options.default_float, options.default_int) == ("float32", "int32")
    options = DoccOptions(default_float="float64", default_int="int64")
    assert (options.default_float, options.default_int) == ("float64", "int64")
