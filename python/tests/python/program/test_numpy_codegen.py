import numpy as np
import pytest

from docc.python import native, register_target_overrides


def _axpy_sqrt(A, B, alpha):
    return alpha * A + np.sqrt(B)


def _matmul(A, B):
    return A @ B


def _column_sum(A):
    return np.sum(A, axis=0)


def _int_div(A, B):
    return A / B


def _reference(kernel, *args):
    if kernel is _int_div:
        # Integer `/` lowers to C division, which truncates toward zero.
        A, B = args
        return np.sign(A) * np.sign(B) * (np.abs(A) // np.abs(B))
    return kernel(*args)


def _inputs(kernel):
    rng = np.random.default_rng(0)
    if kernel is _axpy_sqrt:
        return rng.random(33), rng.random(33), 1.5
    if kernel is _matmul:
        return rng.random((7, 5)), rng.random((5, 9))
    if kernel is _column_sum:
        return (rng.random((6, 11)),)
    divisors = rng.choice([-7, -3, -1, 2, 5], size=17).astype(np.int64)
    return rng.integers(-50, 50, size=17).astype(np.int64), divisors


@pytest.mark.parametrize("target", ["none", "sequential"])
@pytest.mark.parametrize("kernel", [_axpy_sqrt, _matmul, _column_sum, _int_div])
def test_numpy_codegen_matches_cpp(kernel, target):
    args = _inputs(kernel)

    numpy_program = native(kernel, target=target, codegen="numpy")
    compiled = numpy_program.compile(*args)
    assert compiled.lib_path.endswith(".py")

    expected = native(kernel, target=target, codegen="cpp")(*args)
    np.testing.assert_allclose(numpy_program(*args), expected)
    np.testing.assert_allclose(expected, _reference(kernel, *args))


def test_numpy_codegen_rejects_unsupported_target():
    program = native(_axpy_sqrt, target="openmp", codegen="numpy")
    with pytest.raises(ValueError, match="does not support codegen 'numpy'"):
        program.compile(*_inputs(_axpy_sqrt))


def test_registered_target_receives_codegen():
    seen = {}

    def _schedule(sdfg, category, options):
        pass

    def _compile(sdfg, output_folder, instrumentation_mode, capture_args, options):
        seen["codegen"] = options["codegen"]
        return sdfg._compile(
            output_folder,
            "none",
            instrumentation_mode,
            capture_args,
            codegen=options["codegen"],
        )

    register_target_overrides("numpy_only", _schedule, _compile, codegens={"numpy"})

    args = _inputs(_axpy_sqrt)
    result = native(_axpy_sqrt, target="numpy_only", codegen="numpy")(*args)
    assert seen["codegen"] == "numpy"
    np.testing.assert_allclose(result, _axpy_sqrt(*args))

    with pytest.raises(ValueError, match="does not support codegen 'cpp'"):
        native(_axpy_sqrt, target="numpy_only", codegen="cpp").compile(*args)
