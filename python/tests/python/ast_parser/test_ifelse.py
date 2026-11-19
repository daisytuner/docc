import pytest

import numpy as np
from docc.python import native


def test_if():
    @native
    def if_test(a) -> int:
        if a > 10:
            return 1
        return 0

    assert if_test(20) == 1
    assert if_test(5) == 0


def test_if_else():
    @native
    def if_else_test(a) -> int:
        if a > 10:
            return 1
        else:
            return 0

    assert if_else_test(20) == 1
    assert if_else_test(5) == 0


def test_if_elif_else():
    @native
    def if_elif_else_test(a) -> int:
        if a > 10:
            return 1
        elif a == 10:
            return 2
        else:
            return 0

    assert if_elif_else_test(20) == 1
    assert if_elif_else_test(10) == 2
    assert if_elif_else_test(5) == 0


def test_nested_if():
    @native
    def nested_if_test(a, b) -> int:
        if a > 10:
            if b > 5:
                return 1
            else:
                return 2
        else:
            return 0

    assert nested_if_test(20, 6) == 1
    assert nested_if_test(20, 4) == 2
    assert nested_if_test(5, 6) == 0
    assert nested_if_test(5, 4) == 0


def test_raise_is_noop():
    # `raise` statements are ignored by the frontend (exceptions are not
    # modelled in the SDFG IR). The exception expression must not be lowered,
    # and the surrounding control flow keeps working.
    @native
    def guarded(a) -> int:
        if a < 0:
            raise ValueError("negative")
        return a * 2

    assert guarded(5) == 10
    assert guarded(3) == 6


def test_raise_bare_is_noop():
    # A bare `raise` (re-raise) is also a no-op.
    @native
    def maybe(a) -> int:
        if a > 100:
            raise
        return a + 1

    assert maybe(10) == 11


def test_conditional_expression_keeps_its_branches_type():
    # The comparison's temporary must not take the conditional's name.
    @native
    def pick(a, out):
        first = 0
        for j in range(1, 4):
            if a[j] > a[first]:
                first = j
        out[0] = 5 if first == 0 else 2
        out[1] = 0.5 if a[first] > 2.0 else 1

    out = np.zeros(2)
    pick(np.array([3.0, 1.0, 2.0, 0.0]), out)
    np.testing.assert_array_equal(out, [5, 0.5])


def test_variables_named_like_math_constants():
    @native
    def skip_one(a, out):
        for e in range(4):
            if e != 2:
                out[e] = a[e]

    out = np.zeros(4)
    skip_one(np.arange(4.0), out)
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 3.0])


@pytest.mark.parametrize("a", [[1.0, 2.0], [np.nan, 2.0], [1.0, np.nan], [3.0, 3.0]])
def test_float_comparisons_follow_ieee_with_nan(a):
    @native
    def compare(a, out):
        out[0] = 1.0 if a[0] < a[1] else 0.0
        out[1] = 1.0 if a[0] >= a[1] else 0.0
        out[2] = 1.0 if a[0] != a[1] else 0.0
        out[3] = 1.0 if a[0] == a[1] else 0.0

    x = np.array(a)
    out = np.full(4, -1.0)
    compare(x, out)
    expected = [
        float(x[0] < x[1]),
        float(x[0] >= x[1]),
        float(x[0] != x[1]),
        float(x[0] == x[1]),
    ]
    np.testing.assert_array_equal(out, expected)
