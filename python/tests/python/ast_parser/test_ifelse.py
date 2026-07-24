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
