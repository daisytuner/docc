import numpy as np
import ml_dtypes
import pytest

from docc.python import native


def test_bfloat16_add():
    """Test bfloat16 addition."""

    @native
    def add_bf16(a, b):
        return a + b

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=ml_dtypes.bfloat16)
    result = add_bf16(a, b)
    expected = a + b

    assert result.dtype == ml_dtypes.bfloat16
    assert result.shape == (4,)
    assert result.strides == (2,)  # bfloat16 is 2 bytes
    assert np.array_equal(result, expected)


def test_bfloat16_mul():
    """Test bfloat16 multiplication."""

    @native
    def mul_bf16(a, b):
        return a * b

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    b = np.array([2.0, 3.0, 4.0, 5.0], dtype=ml_dtypes.bfloat16)
    result = mul_bf16(a, b)
    expected = a * b

    assert result.dtype == ml_dtypes.bfloat16
    assert result.shape == (4,)
    assert np.array_equal(result, expected)
