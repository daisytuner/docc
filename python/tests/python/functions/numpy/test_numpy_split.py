import sys

from docc.python import native
import numpy as np
import pytest


def test_split_axis0_1d():
    """np.split of a 1-D array into equal sections along axis 0."""

    @native
    def split1d(a):
        p, q = np.split(a, 2)
        return p, q

    a = np.arange(8, dtype=np.float64)
    p, q = split1d(a.copy())
    ep, eq = np.split(a, 2)
    assert p.shape == ep.shape and q.shape == eq.shape
    assert np.array_equal(p, ep)
    assert np.array_equal(q, eq)


def test_split_axis1_columns():
    """np.split along axis 1 into single-column sections (LULESH pattern)."""

    @native
    def split_cols(a):
        x0, x1, x2 = np.split(a, 3, axis=1)
        return x0, x1, x2

    a = np.arange(12, dtype=np.float64).reshape(4, 3)
    res = split_cols(a.copy())
    exp = np.split(a, 3, axis=1)
    for r, e in zip(res, exp):
        assert r.shape == e.shape
        assert np.array_equal(r, e)


def test_split_axis1_blocks():
    """np.split along axis 1 into multi-column blocks."""

    @native
    def split_blocks(a):
        lo, hi = np.split(a, 2, axis=1)
        return lo, hi

    a = np.arange(16, dtype=np.float64).reshape(2, 8)
    lo, hi = split_blocks(a.copy())
    elo, ehi = np.split(a, 2, axis=1)
    assert lo.shape == elo.shape and hi.shape == ehi.shape
    assert np.array_equal(lo, elo)
    assert np.array_equal(hi, ehi)


def test_split_negative_axis():
    """np.split accepts a negative axis (-1 == last axis)."""

    @native
    def split_neg(a):
        u, v = np.split(a, 2, axis=-1)
        return u, v

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    u, v = split_neg(a.copy())
    eu, ev = np.split(a, 2, axis=-1)
    assert np.array_equal(u, eu)
    assert np.array_equal(v, ev)


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
def test_split_values_used_in_computation():
    """Split sections can be consumed in arithmetic, not just returned."""

    @native
    def split_sum(a, out):
        x0, x1, x2 = np.split(a, 3, axis=1)
        out[:] = x0[:, 0] + x1[:, 0] + x2[:, 0]

    a = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = np.zeros(4, dtype=np.float64)
    split_sum(a.copy(), out)
    x0, x1, x2 = np.split(a, 3, axis=1)
    expected = x0[:, 0] + x1[:, 0] + x2[:, 0]
    assert np.allclose(out, expected)


def test_split_list_indices_unsupported():
    """Splitting at an explicit list of indices is not supported and fails clearly."""

    @native
    def split_list(a):
        parts = np.split(a, [1, 3], axis=1)
        return parts

    a = np.arange(12, dtype=np.float64).reshape(4, 3)
    with pytest.raises(NotImplementedError):
        split_list.compile(a)


def test_split_uneven_constant_unsupported():
    """A compile-time-constant uneven split fails clearly."""

    @native
    def split_uneven(a):
        parts = np.split(np.reshape(a, (2, 3)), 2, axis=1)
        return parts

    a = np.arange(6, dtype=np.float64)
    with pytest.raises(NotImplementedError):
        split_uneven.compile(a)
