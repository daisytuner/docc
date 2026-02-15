import pytest

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
