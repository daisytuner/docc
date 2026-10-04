import itertools

import pytest

from docc.compiler.compiled_sdfg import idiv, imod


def c_div(a, b):
    # C truncates toward zero.
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


@pytest.mark.parametrize(
    "a, b",
    list(itertools.product(range(-17, 18), [-8, -5, -3, -2, -1, 1, 2, 3, 5, 8])),
)
def test_shape_evaluation_matches_c(a, b):
    assert idiv(a, b) == c_div(a, b)
    assert imod(a, b) == a - b * c_div(a, b)
    assert b * idiv(a, b) + imod(a, b) == a


def test_known_values():
    assert idiv(-7, 2) == -3
    assert imod(-7, 2) == -1
    assert idiv(7, -2) == -3
    assert imod(7, -2) == 1
    assert idiv(-7, -2) == 3
    assert imod(-7, -2) == -1
