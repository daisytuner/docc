import pytest

from docc.python import native


def test_for_range():
    @native
    def for_range(n) -> int:
        s = 0
        for i in range(n):
            s = s + i
        return s

    assert for_range(10) == 45


def test_for_range_positive():
    @native
    def for_range_positive(n):
        s = 0
        for i in range(0, n, 1):
            s = s + i
        return s

    s = for_range_positive(10)
    assert s == 45


def test_for_range_positive_strided():
    @native
    def for_range_positive_strided(n):
        s = 0
        for i in range(0, n, 2):
            s = s + i
        return s

    s = for_range_positive_strided(10)
    assert s == 20


def test_for_range_negative():
    @native
    def for_range_negative(n):
        s = 0
        for i in range(n - 1, -1, -1):
            s = s + i
        return s

    assert for_range_negative(10) == 45


def test_for_range_two_args():
    @native
    def for_range_two_args(start, stop) -> int:
        s = 0
        for i in range(start, stop):
            s = s + i
        return s

    assert for_range_two_args(5, 10) == 35  # 5+6+7+8+9
    assert for_range_two_args(5, 1) == 0  # empty range


def test_for_nested():
    @native
    def for_nested(n, m) -> int:
        s = 0
        for i in range(n):
            for j in range(m):
                s = s + i * m + j
        return s

    assert for_nested(3, 4) == 66  # sum of 0..11


@pytest.mark.skip(
    reason="fors are created as structured loops, which don't support break statements"
)
def test_for_with_break():
    @native
    def for_with_break(n) -> int:
        s = 0
        for i in range(n):
            if i == 5:
                break
            s = s + i
        return s

    assert for_with_break(10) == 10  # 0+1+2+3+4


@pytest.mark.skip(
    reason="fors are created as structured loops, which don't support continue statements"
)
def test_for_with_continue():
    @native
    def for_with_continue(n) -> int:
        s = 0
        for i in range(n):
            if i % 2 == 1:
                continue
            s = s + i
        return s

    assert for_with_continue(10) == 20  # 0+2+4+6+8


def test_for_range_expression_args():
    @native
    def for_range_expression_args(n) -> int:
        s = 0
        for i in range(n * 2):
            s = s + 1
        return s

    assert for_range_expression_args(5) == 10


def test_for_unused_variable():
    @native
    def for_unused_variable(n) -> int:
        s = 0
        for _ in range(n):
            s = s + 1
        return s

    assert for_unused_variable(7) == 7
