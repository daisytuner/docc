from docc.python import native
import pytest
import numpy as np


def test_int_max():
    @native
    def int_max(a, b):
        return max(a, b)

    compiled = int_max.compile(1, 2)
    res = compiled(10, 20)
    assert res == 20
    res = compiled(20, 10)
    assert res == 20


def test_int_min():
    @native
    def int_min(a, b):
        return min(a, b)

    compiled = int_min.compile(1, 2)
    res = compiled(10, 20)
    assert res == 10
    res = compiled(20, 10)
    assert res == 10


def test_float_max():
    @native
    def float_max(a, b):
        return max(a, b)

    compiled = float_max.compile(1.0, 2.0)
    res = compiled(10.5, 20.5)
    assert res == 20.5
    res = compiled(20.5, 10.5)
    assert res == 20.5


def test_float_min():
    @native
    def float_min(a, b):
        return min(a, b)

    compiled = float_min.compile(1.0, 2.0)
    res = compiled(10.5, 20.5)
    assert res == 10.5
    res = compiled(20.5, 10.5)
    assert res == 10.5


def test_mixed_max():
    @native
    def mixed_max(a, b):
        return max(a, b)

    compiled = mixed_max.compile(1, 2.0)
    res = compiled(10, 20.5)
    assert res == 20.5
    res = compiled(30, 20.5)
    assert res == 30.0
    assert isinstance(res, float)


def test_mixed_min():
    @native
    def mixed_min(a, b):
        return min(a, b)

    compiled = mixed_min.compile(1, 2.0)
    res = compiled(10, 20.5)
    assert res == 10.0
    assert isinstance(res, float)
    res = compiled(30, 20.5)
    assert res == 20.5
