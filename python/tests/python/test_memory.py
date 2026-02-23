"""Tests for memory management: malloc hoisting and free emission."""

from typing import Annotated

import numpy as np

from docc.python import native
from docc.python.memory import ManagedMemoryHandler
from docc.sdfg import Pointer, Scalar, PrimitiveType


def test_simple_alloc_hoisted():
    """Allocation size depends only on function arg - should be hoisted."""

    @native(target="none")
    def simple_alloc(n: int):
        a = np.empty(n, dtype=np.float64)
        a[0] = 1.0
        return a

    result = simple_alloc(10)
    assert result.shape == (10,)
    assert result[0] == 1.0

    stats = simple_alloc.last_sdfg.loop_report()
    assert stats["Malloc"] == 1
    assert stats["Free"] == 1


def test_zeros_hoisted_with_memset():
    """np.zeros should hoist malloc and memset."""

    @native(target="none")
    def zeros_alloc(n: int):
        a = np.zeros(n, dtype=np.float64)
        return a

    result = zeros_alloc(5)
    assert result.shape == (5,)
    assert np.all(result == 0.0)

    stats = zeros_alloc.last_sdfg.loop_report()
    assert stats["Malloc"] == 1
    assert stats["Memset"] == 1
    assert stats["Free"] == 1


def test_ones_not_hoisted():
    """np.ones requires loop init - should NOT be hoisted."""

    @native(target="none")
    def ones_alloc(n: int):
        a = np.ones(n, dtype=np.float64)
        return a

    result = ones_alloc(5)
    assert result.shape == (5,)
    assert np.all(result == 1.0)

    stats = ones_alloc.last_sdfg.loop_report()
    assert stats["Malloc"] == 1
    assert stats.get("Free", 0) == 0  # return value, not freed


def test_free_before_return():
    """Temporary allocation should have free before return."""

    @native(target="none")
    def sum_with_temp(x: Annotated[np.ndarray, "N", np.float64]):
        temp = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            temp[i] = x[i] * 2
        return temp

    x = np.array([1.0, 2.0, 3.0])
    result = sum_with_temp(x)
    assert result.shape == (3,)
    assert np.allclose(result, [2.0, 4.0, 6.0])

    stats = sum_with_temp.last_sdfg.loop_report()
    assert stats["Malloc"] == 1
    assert stats["Memset"] == 1
    assert stats["Free"] == 1


def test_multiple_allocs_multiple_frees():
    """Multiple allocations should all be freed."""

    @native(target="none")
    def multi_alloc(x: Annotated[np.ndarray, "N", np.float64]):
        a = np.zeros(x.shape[0], dtype=np.float64)
        b = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            a[i] = x[i]
            b[i] = x[i] * 2
        return a

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = multi_alloc(x)
    assert result.shape == (5,)
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    stats = multi_alloc.last_sdfg.loop_report()
    assert stats["Malloc"] == 2
    assert stats["Memset"] == 2
    assert stats["Free"] == 2


def test_sum_reduction_hoisted():
    """np.sum creates temp array - should be hoisted."""

    @native(target="none")
    def sum_rows(x: Annotated[np.ndarray, "N,M", np.float32]):
        return np.sum(x, axis=-1, keepdims=True)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = sum_rows(x)
    assert result.shape == (2, 1)
    assert np.allclose(result, [[3.0], [7.0]])

    stats = sum_rows.last_sdfg.loop_report()
    assert stats["Malloc"] == 1
    assert stats["Free"] == 1


def test_allocate_checks_hoistability():
    """Verify allocate() returns True/False based on hoistability."""

    class MockBuilderHoistable:
        def is_hoistable_size(self, size_expr):
            return True

    class MockBuilderNotHoistable:
        def is_hoistable_size(self, size_expr):
            return False

    handler1 = ManagedMemoryHandler(MockBuilderHoistable())
    result1 = handler1.allocate("a", Pointer(Scalar(PrimitiveType.Double)), "8*n")
    assert result1 is True
    assert handler1.has_allocations()

    handler2 = ManagedMemoryHandler(MockBuilderNotHoistable())
    result2 = handler2.allocate("b", Pointer(Scalar(PrimitiveType.Double)), "8*i")
    assert result2 is False
    assert not handler2.has_allocations()
