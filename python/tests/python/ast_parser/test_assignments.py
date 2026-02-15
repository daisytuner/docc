from docc.python import native
import numpy as np


def test_tuple_unpacking():
    @native
    def unpack_two_values() -> int:
        a, b = 1, 2
        return a + b

    assert unpack_two_values() == 3

    @native
    def unpack_three_values() -> int:
        a, b, c = 1, 2, 3
        return a + b + c

    assert unpack_three_values() == 6

    @native
    def unpack_expressions(a, b) -> int:
        d, e = a + 1, b * 2
        return d + e

    assert unpack_expressions(2, 3) == 9


def test_multiple_assignment():
    @native
    def double_assign(in_val: int) -> int:
        a = b = in_val
        return a + b

    assert double_assign(5) == 10

    @native
    def double_assign_expression(in_val: int) -> int:
        a = b = in_val + 2
        return a * b

    assert double_assign_expression(3) == 25

    @native
    def triple_assign(val: int) -> int:
        a = b = c = val
        return a + b + c

    assert triple_assign(10) == 30
