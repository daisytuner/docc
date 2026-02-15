import sys
import os
import pytest
import numpy as np
import docc.sdfg

from docc.python import native


def test_simple_scalars():
    @native
    def scalar_func(a, b, c):
        pass

    # Trigger build with sample arguments
    compiled = scalar_func.compile(1.0, 2, True)
    sdfg = scalar_func.last_sdfg

    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)
    assert sdfg.type("a").primitive_type == docc.sdfg.PrimitiveType.Double
    assert sdfg.type("b").primitive_type == docc.sdfg.PrimitiveType.Int64
    assert sdfg.type("c").primitive_type == docc.sdfg.PrimitiveType.Bool


def test_numpy_scalars():
    @native
    def numpy_scalar_func(a, b, c):
        pass

    # Trigger build
    compiled = numpy_scalar_func.compile(np.float32(1.0), np.int32(2), np.float64(3.0))
    sdfg = numpy_scalar_func.last_sdfg

    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)
    assert sdfg.type("a").primitive_type == docc.sdfg.PrimitiveType.Float
    assert sdfg.type("b").primitive_type == docc.sdfg.PrimitiveType.Int32
    assert sdfg.type("c").primitive_type == docc.sdfg.PrimitiveType.Double


def test_arrays_runtime():
    @native
    def array_func(A, B):
        pass

    # Trigger build with arrays
    A_arr = np.zeros(10, dtype=np.float32)
    B_arr = np.zeros(20, dtype=np.int32)
    compiled = array_func.compile(A_arr, B_arr)
    sdfg = array_func.last_sdfg

    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)
    assert isinstance(sdfg.type("A"), docc.sdfg.Pointer)
    assert sdfg.type("A").pointee_type.primitive_type == docc.sdfg.PrimitiveType.Float


def test_arrays_multidim():
    @native
    def multidim_func(A):
        pass

    # Trigger build with multidim array
    A_arr = np.zeros((10, 20), dtype=np.float64)
    compiled = multidim_func.compile(A_arr)
    sdfg = multidim_func.last_sdfg

    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)
    assert isinstance(sdfg.type("A"), docc.sdfg.Pointer)
    assert sdfg.type("A").pointee_type.primitive_type == docc.sdfg.PrimitiveType.Double


def test_mixed_arguments():
    @native
    def mixed_func(N, A):
        pass

    # Trigger build
    A_arr = np.zeros(10, dtype=np.float32)
    compiled = mixed_func.compile(10, A_arr)
    sdfg = mixed_func.last_sdfg

    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)


def test_invalid_annotation():
    # This test is less relevant now as we don't check annotations,
    # but we can check if passing an unsupported type raises error
    @native
    def invalid_type(a):
        pass

    with pytest.raises(ValueError, match="Unsupported argument type"):
        invalid_type.compile("string")  # str is not supported yet
