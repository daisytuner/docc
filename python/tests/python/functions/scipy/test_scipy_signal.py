from docc.python import native
import pytest
import numpy as np
import scipy.signal


@pytest.mark.skip()
def test_scipy_correlate2d_valid():
    @native
    def correlate2d_valid(n: int, k: int) -> float:
        a = np.zeros((n, n), dtype=float)
        b = np.zeros((k, k), dtype=float)
        # Initialize some values
        a[0, 0] = 1.0
        a[0, 1] = 2.0
        b[0, 0] = 1.0

        c = scipy.signal.correlate2d(a, b, mode="valid")
        return c[0, 0]

    # Verify with numpy/scipy execution
    n = 5
    k = 3
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((k, k), dtype=float)
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    b[0, 0] = 1.0
    expected = scipy.signal.correlate2d(a, b, mode="valid")[0, 0]

    assert correlate2d_valid(n, k) == expected


@pytest.mark.skip()
def test_scipy_correlate2d_same():
    @native
    def correlate2d_same(n: int, k: int) -> float:
        a = np.zeros((n, n), dtype=float)
        b = np.zeros((k, k), dtype=float)
        # Initialize some values
        a[1, 1] = 1.0
        b[1, 1] = 2.0

        c = scipy.signal.correlate2d(a, b, mode="same")
        return c[1, 1]

    # Verify with numpy/scipy execution
    n = 5
    k = 3
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((k, k), dtype=float)
    a[1, 1] = 1.0
    b[1, 1] = 2.0
    expected = scipy.signal.correlate2d(a, b, mode="same")[1, 1]

    assert correlate2d_same(n, k) == expected


@pytest.mark.skip(
    reason="Full mode test currently disabled due to handling of boundary conditions."
)
def test_scipy_correlate2d_full():
    @native
    def correlate2d_full(n: int, k: int) -> float:
        a = np.zeros((n, n), dtype=float)
        b = np.zeros((k, k), dtype=float)
        # Initialize some values
        a[0, 0] = 1.0
        a[n - 1, n - 1] = 2.0
        b[0, 0] = 3.0

        c = scipy.signal.correlate2d(a, b, mode="full")
        return c[0, 0]

    # Verify with numpy/scipy execution
    n = 5
    k = 3
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((k, k), dtype=float)
    a[0, 0] = 1.0
    a[n - 1, n - 1] = 2.0
    b[0, 0] = 3.0
    expected = scipy.signal.correlate2d(a, b, mode="full")[0, 0]

    assert correlate2d_full(n, k) == expected
