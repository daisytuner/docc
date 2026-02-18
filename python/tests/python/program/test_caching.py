import numpy as np
from docc.python import native
import pytest


def test_caching_behavior():
    @native
    def simple_add(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    # 1. First compilation (N=1024)
    N = 1024
    A = np.random.rand(N)
    B = np.random.rand(N)
    C = np.zeros(N)

    compiled_1 = simple_add.compile(A, B, C)

    # 2. Second compilation (N=2048) - Should hit cache
    N2 = 2048
    A2 = np.random.rand(N2)
    B2 = np.random.rand(N2)
    C2 = np.zeros(N2)

    compiled_2 = simple_add.compile(A2, B2, C2)

    # Check if they are the exact same object
    assert compiled_1 is compiled_2
    print("Cache hit confirmed for same shape structure but different values.")

    # Verify execution works with new size
    compiled_2(A2, B2, C2)
    assert np.allclose(C2, A2 + B2)

    # 3. Third compilation (Mismatch sizes) - Should recompile
    # Note: This specific kernel might fail at runtime if we don't handle bounds,
    # but here we just check if it triggers a new compilation.
    # We'll use a kernel that supports different sizes to be safe.

    @native
    def flexible_kernel(A, B):
        # Just do something safe
        pass

    N = 100
    A = np.zeros(N)
    B = np.zeros(N)

    c1 = flexible_kernel.compile(A, B)  # A=N, B=N -> _s0, _s0

    M = 50
    B_diff = np.zeros(M)
    c2 = flexible_kernel.compile(A, B_diff)  # A=N, B=M -> _s0, _s1

    assert c1 is not c2
    print("Recompilation confirmed for different shape structure.")
