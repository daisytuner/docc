import pytest

from docc.python import native


def test_while_false():
    @native
    def while_test_false(counter) -> int:
        a = 0
        while False:
            a = a + 1
        return a

    assert while_test_false(3) == 0


def test_while():
    @native
    def while_test(counter) -> int:
        a = 0
        while counter < 5:
            counter = counter + 1
            a = a + 1

        return a

    assert while_test(0) == 5
    assert while_test(3) == 2
    assert while_test(5) == 0
    assert while_test(10) == 0


def test_while_break():
    @native
    def while_test_break(counter) -> int:
        a = 0
        while counter < 5:
            if counter == 3:
                break
            counter = counter + 1
            a = a + 1

        return a

    assert while_test_break(0) == 3
    assert while_test_break(2) == 1
    assert while_test_break(3) == 0
    assert while_test_break(4) == 1


def test_while_continue():
    @native
    def while_test_continue(counter) -> int:
        a = 0
        while counter < 5:
            counter = counter + 1
            if counter == 3:
                continue
            a = a + 1

        return a

    assert while_test_continue(0) == 4
    assert while_test_continue(2) == 2
    assert while_test_continue(3) == 2
    assert while_test_continue(4) == 1


def test_while_true():
    @native
    def while_true(limit) -> int:
        a = 0
        while True:
            a = a + 1
            if a >= limit:
                break
        return a

    assert while_true(1) == 1
    assert while_true(5) == 5
    assert while_true(10) == 10


def test_while_condition_and():
    @native
    def while_and(a, b) -> int:
        count = 0
        while a < 5 and b < 5:
            a = a + 1
            b = b + 1
            count = count + 1
        return count

    assert while_and(0, 0) == 5
    assert while_and(3, 0) == 2
    assert while_and(0, 4) == 1
    assert while_and(5, 0) == 0


def test_while_condition_or():
    @native
    def while_or(a, b) -> int:
        count = 0
        while a < 3 or b < 3:
            if a < 3:
                a = a + 1
            if b < 3:
                b = b + 1
            count = count + 1
        return count

    assert while_or(0, 0) == 3
    assert while_or(2, 0) == 3
    assert while_or(0, 2) == 3
    assert while_or(3, 3) == 0


def test_while_condition_not():
    @native
    def while_not(flag, counter) -> int:
        count = 0
        while not flag:
            count = count + 1
            counter = counter + 1
            if counter >= 5:
                flag = True
        return count

    assert while_not(False, 0) == 5
    assert while_not(False, 3) == 2
    assert while_not(True, 0) == 0


def test_while_nested():
    @native
    def while_nested(n) -> int:
        total = 0
        i = 0
        while i < n:
            j = 0
            while j < n:
                total = total + 1
                j = j + 1
            i = i + 1
        return total

    assert while_nested(0) == 0
    assert while_nested(1) == 1
    assert while_nested(3) == 9
    assert while_nested(5) == 25


def test_while_nested_break():
    @native
    def while_nested_break(n) -> int:
        total = 0
        i = 0
        while i < n:
            j = 0
            while j < n:
                if j == 2:
                    break
                total = total + 1
                j = j + 1
            i = i + 1
        return total

    assert while_nested_break(1) == 1
    assert while_nested_break(3) == 6
    assert while_nested_break(5) == 10
