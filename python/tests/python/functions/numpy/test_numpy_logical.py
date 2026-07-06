import numpy as np

from docc.python import native


def test_bitwise_array_and():
    """Elementwise integer bitwise AND on arrays (LULESH bc_mask & mask)."""

    @native
    def arr_and(a, b, out):
        out[:] = a & b

    a = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    b = np.array([0x1, 0x2, 0x38, 0x0], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    arr_and(a.copy(), b.copy(), out)
    assert np.array_equal(out, a & b)


def test_bitwise_array_or():
    """Elementwise integer bitwise OR on arrays."""

    @native
    def arr_or(a, b, out):
        out[:] = a | b

    a = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    b = np.array([0x1, 0x2, 0x38, 0x0], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    arr_or(a.copy(), b.copy(), out)
    assert np.array_equal(out, a | b)


def test_bitwise_array_xor():
    """Elementwise integer bitwise XOR on arrays."""

    @native
    def arr_xor(a, b, out):
        out[:] = a ^ b

    a = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    b = np.array([0x1, 0x2, 0x38, 0x0], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    arr_xor(a.copy(), b.copy(), out)
    assert np.array_equal(out, a ^ b)


def test_bitwise_array_scalar_and():
    """Bitwise AND between an integer array and a scalar constant."""

    @native
    def and_scalar(a, out):
        out[:] = a & 0x7

    a = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    and_scalar(a.copy(), out)
    assert np.array_equal(out, a & 0x7)


def test_bitwise_array_shift():
    """Left shift between an integer array and a scalar constant."""

    @native
    def shift_scalar(a, out):
        out[:] = a << 2

    a = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    shift_scalar(a.copy(), out)
    assert np.array_equal(out, a << 2)
