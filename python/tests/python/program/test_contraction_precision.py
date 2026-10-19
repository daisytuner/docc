import numpy as np
import pytest

from docc.compiler.docc_program import DoccOptions
from docc.python import native


def _matmul(A, B):
    return A @ B


def _inputs(dtype):
    rng = np.random.default_rng(0)
    # Rows and columns of different magnitudes; the emulation scales each separately.
    A = rng.standard_normal((24, 40)) * np.logspace(-3, 3, 24)[:, None]
    B = rng.standard_normal((40, 16)) * np.logspace(2, -2, 16)[None, :]
    return A.astype(dtype), B.astype(dtype)


def _relative_error(result, A, B):
    exact = A.astype(np.longdouble) @ B.astype(np.longdouble)
    magnitude = np.abs(A.astype(np.longdouble)) @ np.abs(B.astype(np.longdouble))
    return float(np.max(np.abs(result - exact) / magnitude))


def _run(precision, dtype, target="none"):
    A, B = _inputs(dtype)
    # The emulation needs a constant reduction length to bound its Int32 sums.
    program = native(
        _matmul, target=target, contraction_precision=precision, symbolic_shapes=False
    )
    return _relative_error(program(A, B), A, B)


@pytest.mark.parametrize("target", ["none", "sequential"])
@pytest.mark.parametrize("precision", ["ozaki", "ozaki1", "ozaki2"])
def test_ozaki_matches_native_double(precision, target):
    native_error = _run("native", np.float64, target)
    assert _run(precision, np.float64, target) <= max(4 * native_error, 1e-15)


@pytest.mark.parametrize("precision", ["ozaki1", "ozaki2"])
def test_ozaki_matches_native_float(precision):
    assert _run(precision, np.float32) < 1e-6


def test_int8_is_within_quantization_error():
    assert _run("int8", np.float32) < 1e-2


def test_fewer_moduli_lose_accuracy():
    assert _run("ozaki2:4", np.float64) > 1e-10


@pytest.mark.parametrize("precision", ["int9", "ozaki3", "fp8", "ozaki:4"])
def test_invalid_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="contraction_precision"):
        DoccOptions(contraction_precision=precision)
