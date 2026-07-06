from docc.python import native
import numpy as np
import pytest


def helper_nested(a):
    return a * 2


def helper_outer(a):
    return helper_nested(a) + 1


def test_inlining():
    @native
    def inlining(a):
        return helper_nested(a)

    assert inlining(10) == 20


def test_nested_inlining():
    @native
    def nested_inlining(a):
        return helper_outer(a)

    assert nested_inlining(10) == 21


def helper_with_local_var(a):
    b = 10
    return a + b


def test_inlining_local_vars():
    @native
    def inlining_local_vars(a):
        b = 5
        return helper_with_local_var(a) + b

    assert inlining_local_vars(20) == 35  # (20 + 10) + 5


def test_inlining_local_function():
    def helper_nested(a):
        return a * 4

    @native
    def inlining_local_function(a):
        return helper_nested(a)

    assert inlining_local_function(10) == 40


def helper_add(a, b):
    return a + b


def test_inlining_multiple_calls():
    @native
    def multiple_calls(a):
        x = helper_nested(a)
        y = helper_nested(a)
        return x + y

    assert multiple_calls(5) == 20  # (5*2) + (5*2)


def test_inlining_multiple_args():
    @native
    def inlining_multiple_args(a, b):
        return helper_add(a, b)

    assert inlining_multiple_args(3, 7) == 10


def test_inlining_captured_constant():
    factor = 3

    def multiply_by_factor(a):
        return a * factor

    @native
    def inlining_captured_constant(a):
        return multiply_by_factor(a)

    assert inlining_captured_constant(10) == 30


def test_inlining_nested_local_functions():
    def inner(a):
        return a + 1

    def outer(a):
        return inner(a) * 2

    @native
    def inlining_nested_local(a):
        return outer(a)

    assert inlining_nested_local(4) == 10  # (4+1) * 2


def helper_conditional(a):
    if a > 0:
        return a
    return -a


def test_inlining_conditional_return():
    @native
    def inlining_conditional(a):
        return helper_conditional(a)

    assert inlining_conditional(5) == 5
    assert inlining_conditional(-5) == 5


def test_inlining_multiple_calls_unique_containers():
    def double_it(x):
        return x * 2

    @native
    def chained_calls(a):
        x = double_it(a)
        y = double_it(x)
        z = double_it(y)
        return z

    assert chained_calls(3) == 24  # 3 * 2 * 2 * 2


def _guard_min(a, cur):
    # Early-return guard followed by a fallback return (LULESH time-constraint
    # reduction pattern).
    if a < cur:
        return a
    return cur


def test_inlining_early_return_guard_to_variable():
    """An inlined guard-return whose result is bound to a variable keeps the
    branch value (not the fallback)."""

    @native
    def use_guard(x, cur):
        y = _guard_min(x, cur)
        return y

    assert use_guard(3, 50) == 3
    assert use_guard(80, 50) == 50


def _classify(a):
    # Multiple early returns at the same level.
    if a > 10:
        return 2
    if a > 0:
        return 1
    return 0


def test_inlining_multiple_early_returns():
    """A chain of guard returns selects the first matching branch."""

    @native
    def classify(a):
        return _classify(a)

    assert classify(15) == 2
    assert classify(5) == 1
    assert classify(-1) == 0
