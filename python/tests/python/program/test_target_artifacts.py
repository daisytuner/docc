import numpy as np
import pytest

import docc.compiler.artifacts as artifacts
from docc.python import native

calls = {}


def _build(artifact_path, sdfg_name):
    calls["build"] = sdfg_name


def _load(artifact_path, sdfg_name):
    calls["load"] = sdfg_name

    def run(A, B):
        calls["args"] = (A, B)
        B[:] = 2.0 * A

    return run


@pytest.fixture
def python_module_target(monkeypatch):
    """Declares a Python-module artifact for the sequential target, as a target with its own code generator does in
    C++ (DoccTarget::artifact); the target still builds its C++ library, which the loader ignores.
    """
    calls.clear()
    artifact = {
        "file": "{name}.py",
        "kind": "python_module",
        "loader": f"{__name__}:_load",
        "build": f"{__name__}:_build",
    }
    monkeypatch.setattr(
        artifacts,
        "target_artifact",
        lambda target: artifact if target == "sequential" else None,
    )


def test_builtin_targets_build_shared_libraries():
    assert artifacts.artifact_file("sequential", "f") == "libf.so"
    assert not artifacts.is_python_module("sequential")
    assert artifacts.load_artifact("sequential", "libf.so", "f") is None


def test_names_the_targets_artifact(python_module_target):
    assert artifacts.artifact_file("sequential", "f") == "f.py"
    assert artifacts.artifact_file("openmp", "f") == "libf.so"
    assert artifacts.is_python_module("sequential")


def test_runs_a_python_module_artifact(python_module_target):
    @native(target="sequential", symbolic_shapes=False)
    def double(A, B):
        for i in range(8):
            B[i] = 2.0 * A[i]

    A = np.arange(8, dtype=np.float32)
    B = np.zeros(8, dtype=np.float32)
    double(A, B)

    np.testing.assert_array_equal(B, 2.0 * A)
    assert calls["build"] == calls["load"]
    assert all(isinstance(arg, np.ndarray) for arg in calls["args"])
