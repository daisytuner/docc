import pytest
import numpy as np

from docc.sdfg import (
    PrimitiveType,
    Scalar,
    Pointer,
    Structure,
    Tensor,
)
from docc.python.types import sdfg_type_from_type


def test_numpy_float64():
    result = sdfg_type_from_type(np.float64)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Double


def test_numpy_float32():
    result = sdfg_type_from_type(np.float32)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Float


def test_numpy_int32():
    result = sdfg_type_from_type(np.int32)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Int32


def test_numpy_int16():
    result = sdfg_type_from_type(np.int16)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Int16


def test_numpy_int8():
    result = sdfg_type_from_type(np.int8)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Int8


def test_numpy_uint64():
    result = sdfg_type_from_type(np.uint64)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.UInt64


def test_numpy_uint32():
    result = sdfg_type_from_type(np.uint32)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.UInt32


def test_numpy_uint16():
    result = sdfg_type_from_type(np.uint16)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.UInt16


def test_numpy_uint8():
    result = sdfg_type_from_type(np.uint8)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.UInt8


def test_numpy_bool():
    result = sdfg_type_from_type(np.bool_)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Bool


def test_numpy_ndarray_float64():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.float64]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Double


def test_numpy_ndarray_float32():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.float32]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Float


def test_numpy_ndarray_int32():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.int32]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Int32


def test_numpy_ndarray_int16():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.int16]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Int16


def test_numpy_ndarray_int8():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.int8]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Int8


def test_numpy_ndarray_uint64():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.uint64]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.UInt64


def test_numpy_ndarray_uint32():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.uint32]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.UInt32


def test_numpy_ndarray_uint16():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.uint16]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.UInt16


def test_numpy_ndarray_uint8():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.uint8]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.UInt8


def test_numpy_ndarray_bool():
    result = sdfg_type_from_type(np.ndarray[tuple[int], np.dtype[np.bool_]])
    assert isinstance(result, Pointer)
    assert result.primitive_type == PrimitiveType.Bool
