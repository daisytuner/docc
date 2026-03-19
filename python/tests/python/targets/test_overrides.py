from docc.python import native
import numpy as np
import pytest
import sys
from docc.python import register_target_overrides, register_target
from typing import Callable, Optional, Dict, Any


def _schedule(sdfg, cat: str, kwargs: Dict[str, any]):
    print("hooking scheduling new")
    sdfg.schedule("sequential", cat, False)


def _schedule_legacy(sdfg, cat: str):
    print("hooking scheduling legacy")
    sdfg.schedule("sequential", cat, False)


def _compile(sdfg, out_dir: str, inst_mode: str, capture: bool, kwargs: Dict[str, Any]):
    print("hooking compile")
    return sdfg._compile(out_dir, "sequential", inst_mode, capture)


def _expand(sdfg, cat: str, kwargs: Dict[str, Any]):
    print("hooking expand")
    return sdfg.expand()


def test_python_target_overrides(capsys):
    register_target("special_legacy", _schedule_legacy)
    register_target_overrides("special", _schedule, _compile, _expand)

    @native(target="special", category="server")
    def matmul_etsoc(A, B, C):
        C = A @ B

    N = 8
    A = np.random.rand(N, N).astype(np.float32)
    A.fill(0.5)
    B = np.random.rand(N, N).astype(np.float32)
    B.fill(2)
    C = np.zeros((N, N), dtype=np.float32)

    print("Input: ", A)
    matmul_etsoc(A, B, C)

    captured = capsys.readouterr()
    assert "hooking scheduling new" in captured.out
    assert "hooking compile" in captured.out
    assert "hooking expand" in captured.out
    assert "Target 'special' is not supported" not in captured.out

    print("Result: ", C)
    assert np.allclose(C, A @ B)


def test_python_target_legacy_override(capsys):
    register_target("special_legacy", _schedule_legacy)
    register_target_overrides("special", _schedule, _compile, _expand)

    @native(target="special_legacy", category="server")
    def matmul_etsoc(A, B, C):
        C = A @ B

    N = 8
    A = np.random.rand(N, N).astype(np.float32)
    A.fill(0.5)
    B = np.random.rand(N, N).astype(np.float32)
    B.fill(2)
    C = np.zeros((N, N), dtype=np.float32)

    print("Input: ", A)
    matmul_etsoc(A, B, C)

    captured = capsys.readouterr()
    assert "hooking scheduling legacy" in captured.out
    assert "Target 'special_legacy' is not supported" not in captured.out

    print("Result: ", C)
    assert np.allclose(C, A @ B)
