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


def test_pass_options():
    # `A @ B` lowers to a GEMM library node. Forcing BLAS expansion replaces it
    # with explicit loop nests: an init nest (i, j) plus a matmul nest (i, j, k),
    # so loop analysis should report 5 loops in total.
    @native(target="sequential", **{"library_node_expansion.force_expand": True})
    def matmul_op(a, b):
        return a @ b

    a = np.random.rand(8, 8)
    b = np.random.rand(8, 8)
    res = matmul_op(a, b)
    assert np.allclose(res, a @ b)

    sdfg = matmul_op.last_sdfg
    assert isinstance(sdfg, docc.sdfg.StructuredSDFG)

    analysis = docc.sdfg.AnalysisManager(sdfg)
    loop_analysis = analysis.loop_analysis()

    loops = loop_analysis.loops()
    assert len(loops) == 5

    # Two outermost nests: init (2 loops) + matmul (3 loops).
    outermost = loop_analysis.outermost_loops()
    assert len(outermost) == 2
    total_loops = sum(loop_analysis.loop_info(loop).num_loops for loop in outermost)
    assert total_loops == 5


def test_symbolic_shapes_default():
    # symbolic_shapes defaults to True: array dimensions become `_s{i}` symbol
    # arguments on the SDFG.
    @native
    def add(a, b):
        return a + b

    a = np.zeros((4, 8), dtype=np.float64)
    b = np.zeros((4, 8), dtype=np.float64)
    add.compile(a, b)
    sdfg = add.last_sdfg

    assert add.options.symbolic_shapes is True
    shape_args = [name for name in sdfg.arguments if name.startswith("_s")]
    assert shape_args, "expected symbolic shape arguments in symbolic mode"


def test_symbolic_shapes_disabled_no_symbols():
    # symbolic_shapes=False bakes the concrete integer sizes into the SDFG, so
    # no `_s{i}` symbol arguments are emitted.
    @native(symbolic_shapes=False)
    def add(a, b):
        return a + b

    a = np.zeros((4, 8), dtype=np.float64)
    b = np.zeros((4, 8), dtype=np.float64)
    add.compile(a, b)
    sdfg = add.last_sdfg

    assert add.options.symbolic_shapes is False
    shape_args = [name for name in sdfg.arguments if name.startswith("_s")]
    assert not shape_args, "concrete mode must not emit symbolic shape arguments"


def test_symbolic_shapes_disabled_runtime():
    # Concrete-shape kernels still compile and execute correctly.
    @native(symbolic_shapes=False)
    def add(a, b):
        return a + b

    a = np.random.rand(4, 8)
    b = np.random.rand(4, 8)
    res = add(a, b)
    assert np.allclose(res, a + b)


def test_symbolic_shapes_disabled_no_cache_alias():
    # With concrete shapes baked in, calls with different sizes must not alias to
    # the same cached binary.
    @native(symbolic_shapes=False)
    def add(a, b):
        return a + b

    a1 = np.random.rand(4, 8)
    b1 = np.random.rand(4, 8)
    res1 = add(a1, b1)
    assert np.allclose(res1, a1 + b1)

    a2 = np.random.rand(16, 32)
    b2 = np.random.rand(16, 32)
    res2 = add(a2, b2)
    assert np.allclose(res2, a2 + b2)
