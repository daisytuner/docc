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


def _inputs(n=512):
    rng = np.random.default_rng(0)
    return rng.random(n, dtype=np.float32), rng.random(n, dtype=np.float32)


def _axelera(codegen="axscript", cores=1):
    return native(
        _vadd,
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
