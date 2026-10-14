import os

import numpy as np
import pytest

from docc.python import native

requires_alto = pytest.mark.skipif(
    not os.environ.get("DOCC_AXELERA_PYTHON"),
    reason="DOCC_AXELERA_PYTHON (interpreter with axelera-compiler) is not set",
)


def _vadd(a, b):
    return a + b


def _sum(a):
    return np.sum(a)


def _max(a):
    return np.max(a)


def _min(a):
    return np.min(a)


_REDUCTIONS = {"sum": (_sum, np.sum), "max": (_max, np.max), "min": (_min, np.min)}


def _div(a, b):
    return a / b


def _neg(x):
    return -x


def _abs(x):
    return np.abs(x)


def _sqrt(x):
    return np.sqrt(x)


def _sin(x):
    return np.sin(x)


def _cos(x):
    return np.cos(x)


def _log2(x):
    return np.log2(x)


def _exp2(x):
    return np.exp2(x)


def _exp(x):
    return np.exp(x)


def _log(x):
    return np.log(x)


def _scale(x):
    return x * 2.0


def _one_minus(x):
    return 1.0 - x


def _relu(x):
    return np.maximum(x, 0.0)


def _matmul(a, b):
    return a @ b


def _mlp(x, w1, w2):
    return np.maximum(x @ w1, 0.0) @ w2


# Each op reads "a" in [-1, 1) and/or "b" in [0.5, 1.5), the domain of division, sqrt and the logarithms.
_ELEMENTWISE = {
    "div": (_div, np.divide, ("a", "b")),
    "neg": (_neg, np.negative, ("a",)),
    "abs": (_abs, np.abs, ("a",)),
    "sqrt": (_sqrt, np.sqrt, ("b",)),
    "sin": (_sin, np.sin, ("a",)),
    "cos": (_cos, np.cos, ("a",)),
    "log2": (_log2, np.log2, ("b",)),
    "exp2": (_exp2, np.exp2, ("a",)),
    "exp": (_exp, np.exp, ("a",)),
    "log": (_log, np.log, ("b",)),
    "scale": (_scale, lambda x: x * 2.0, ("a",)),
    "one_minus": (_one_minus, lambda x: 1.0 - x, ("a",)),
    "relu": (_relu, lambda x: np.maximum(x, 0.0), ("a",)),
}


def _op_inputs(names):
    a, b = _inputs()
    values = {"a": a * 2 - 1, "b": b + 0.5}
    return [values[name] for name in names]


def _inputs(n=512):
    rng = np.random.default_rng(0)
    return rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)


def _axelera(codegen="axscript", cores=1, fn=_vadd):
    return native(
        fn,
        target="axelera",
        codegen=codegen,
        symbolic_shapes=False,
        **{"axelera.num_cores": cores},
    )


@requires_alto
@pytest.mark.parametrize("cores", [1, 2])
def test_axscript_vector_add(cores):
    program = _axelera(cores=cores)
    a, b = _inputs()

    compiled = program.compile(a, b)
    assert compiled.lib_path.endswith(".axs.py")
    alto_dir = os.path.join(os.path.dirname(compiled.lib_path), "alto")
    assert os.path.exists(os.path.join(alto_dir, "model.json"))

    # DPU math runs in float18 on hardware; the CPU stand-in matches up to that precision.
    np.testing.assert_allclose(compiled(a, b), a + b, rtol=1e-2)


@pytest.mark.parametrize("cores", [1, 2, 4])
def test_axelera_schedule_runs_as_numpy(cores):
    a, b = _inputs()
    np.testing.assert_allclose(_axelera("numpy", cores)(a, b), a + b, rtol=1e-6)


@requires_alto
@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.parametrize("reduction", sorted(_REDUCTIONS))
def test_axscript_reduction(reduction, cores):
    fn, reference = _REDUCTIONS[reduction]
    a, _ = _inputs()

    compiled = _axelera(cores=cores, fn=fn).compile(a)
    assert compiled.lib_path.endswith(".axs.py")
    np.testing.assert_allclose(compiled(a), reference(a), rtol=1e-2)


@pytest.mark.parametrize("cores", [1, 2, 4])
@pytest.mark.parametrize("reduction", sorted(_REDUCTIONS))
def test_axelera_reduction_runs_as_numpy(reduction, cores):
    fn, reference = _REDUCTIONS[reduction]
    a, _ = _inputs()
    np.testing.assert_allclose(_axelera("numpy", cores, fn)(a), reference(a), rtol=1e-5)


@requires_alto
@pytest.mark.parametrize("op", sorted(_ELEMENTWISE))
def test_axscript_elementwise_op(op):
    fn, reference, names = _ELEMENTWISE[op]
    args = _op_inputs(names)

    compiled = _axelera(fn=fn).compile(*args)
    assert compiled.lib_path.endswith(".axs.py")
    np.testing.assert_allclose(compiled(*args), reference(*args), rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("op", sorted(_ELEMENTWISE))
def test_axelera_elementwise_op_runs_as_numpy(op):
    fn, reference, names = _ELEMENTWISE[op]
    args = _op_inputs(names)
    np.testing.assert_allclose(
        _axelera("numpy", 1, fn)(*args), reference(*args), rtol=1e-5, atol=1e-6
    )


@requires_alto
@pytest.mark.parametrize("cores", [1, 2])
def test_axscript_matmul(cores):
    rng = np.random.default_rng(0)
    a = rng.random((128, 128), dtype=np.float32) * 2 - 1
    b = rng.random((128, 128), dtype=np.float32) * 2 - 1

    compiled = _axelera(cores=cores, fn=_matmul).compile(a, b)
    assert compiled.lib_path.endswith(".axs.py")
    # The MVM multiplies a and b quantized to int8 with one scale per matrix.
    expected = a @ b
    np.testing.assert_allclose(
        compiled(a, b), expected, rtol=0, atol=0.03 * np.abs(expected).max()
    )


@requires_alto
def test_axscript_mlp():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 128), dtype=np.float32)
    w1 = rng.standard_normal((128, 128), dtype=np.float32) / np.sqrt(128)
    w2 = rng.standard_normal((128, 128), dtype=np.float32) / np.sqrt(128)

    expected = _mlp(x, w1, w2)
    np.testing.assert_allclose(
        _axelera(cores=2, fn=_mlp)(x, w1, w2),
        expected,
        rtol=0,
        atol=0.05 * np.abs(expected).max(),
    )


def test_axscript_requires_alto(monkeypatch):
    monkeypatch.delenv("DOCC_AXELERA_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="DOCC_AXELERA_PYTHON"):
        _axelera().compile(*_inputs())


def test_axscript_rejects_partial_pwords():
    with pytest.raises(RuntimeError, match="not whole PWORDs"):
        _axelera().compile(*_inputs(100))


def test_axscript_rejects_uneven_core_split():
    with pytest.raises(RuntimeError, match="do not split"):
        _axelera(cores=3).compile(*_inputs())


def test_axelera_rejects_cpp_codegen():
    program = native(_vadd, target="axelera", codegen="cpp", symbolic_shapes=False)
    with pytest.raises(ValueError, match="does not support codegen 'cpp'"):
        program.compile(*_inputs())
