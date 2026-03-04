from docc.python import native
import numpy as np
import pytest
import sys
from docc.python import register_target_overrides
from typing import Callable, Optional, Dict, Any


def _schedule(sdfg, cat: str, kwargs: Dict[str, any]):
    print("hooking scheduling")
    sdfg.schedule("sequential", cat, False)


def _compile(sdfg, out_dir: str, inst_mode: str, capture: bool, kwargs: Dict[str, Any]):
    print("hooking compile")
    return sdfg._compile(out_dir, "sequential", inst_mode, capture)


register_target_overrides("special", _schedule, _compile)


def test_python_target_overrides(capsys):
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
    assert "hooking scheduling" in captured.out
    assert "hooking compile" in captured.out

    print("Result: ", C)
    assert np.allclose(C, A @ B)
