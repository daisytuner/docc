"""Device-resident argument promotion: full behavioural matrix (numpy frontend).

The promotion pass flips every pointer argument to device-resident storage when
the whole program keeps its data on device, i.e. when every pointer argument is
only ever touched by the offloading boundary nodes (H2D/D2H). When that holds the
compiled artifact expects device pointers and avoids host<->device copies at the
boundary.

The resulting :class:`CompiledSDFG` dispatches on the *call mode* (the kind of
array passed at call time) combined with whether the artifact is device-resident:

    call mode     | non-device-resident       | device-resident
    --------------|---------------------------|------------------------------
    NumPyCPU      | host execution            | host->device copy + warning
    NumPyGPU/cupy | TypeError (rejected)      | zero-copy device execution

Mixing array kinds in a single call is rejected. When a GPU target falls back to
host execution because promotion did not apply, a one-time performance warning is
emitted instead.

Promotion succeeds for a fully data-parallel kernel (all loops are offloaded) and
fails when a pointer argument is used by host code, which we trigger with a
sequential loop carrying a true dependence (cannot be offloaded).

These tests cover the promotion decision, every valid (artifact, call-mode)
combination and its numerical result, the rejected combinations (mixed kinds,
device arrays on a host artifact), and warning emission (and absence) for each
path. All tests require a CUDA device (cupy) and are marked accordingly.
"""

from contextlib import contextmanager
import warnings

from docc.python import native
from docc.compiler import DoccPerformanceWarning
import numpy as np
import pytest


def _parallel_vec_add(target):
    """Fully data-parallel kernel: promotion succeeds (device-resident)."""

    @native(target=target, category="server")
    def vec_add(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    return vec_add


def _sequential_prefix_sum(target):
    # Loop-carried dependence: A[i] depends on A[i - 1], so the loop cannot be
    # parallelized and stays on the host. The pointer argument is therefore used
    # by host code and promotion must be rejected.
    @native(target=target, category="server")
    def prefix_sum(A, N):
        for i in range(1, N):
            A[i] = A[i - 1] + A[i]

    return prefix_sum


@contextmanager
def _no_perf_warning():
    """Assert that no DoccPerformanceWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DoccPerformanceWarning)
        yield


def _compile_vec_add(target="cuda"):
    """Compile the parallel kernel and return (compiled, a, b, N)."""
    N = 1024
    a = np.random.rand(N).astype(np.float64)
    b = np.random.rand(N).astype(np.float64)
    # Build the SDFG from host arrays (shape inference); promotion runs here.
    compiled = _parallel_vec_add(target).compile(a, b, np.zeros(N, dtype=np.float64), N)
    return compiled, a, b, N


def _compile_prefix_sum(target="cuda"):
    """Compile the sequential kernel and return (compiled, a, N)."""
    N = 1024
    a = np.random.rand(N).astype(np.float64)
    compiled = _sequential_prefix_sum(target).compile(a.copy(), N)
    return compiled, a, N


# --------------------------------------------------------------------------- #
# Promotion decision
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_promotion_success_marks_device_resident():
    compiled, *_ = _compile_vec_add("cuda")
    assert compiled.device_resident is True
    assert compiled.device_backend == "cuda"


@pytest.mark.cuda()
def test_promotion_fail_stays_host_resident():
    compiled, *_ = _compile_prefix_sum("cuda")
    assert compiled.device_resident is False
    assert compiled.device_backend is None


# --------------------------------------------------------------------------- #
# Device-resident artifact (vec_add) -- call-mode matrix
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_device_resident_with_device_arrays():
    """NumPyGPU on a device-resident artifact: zero-copy, cupy out, no warning."""
    import cupy as cp

    compiled, a, b, N = _compile_vec_add("cuda")
    A = cp.asarray(a)
    B = cp.asarray(b)
    C = cp.zeros(N, dtype=cp.float64)
    with _no_perf_warning():
        compiled(A, B, C, N)
    cp.cuda.runtime.deviceSynchronize()
    assert isinstance(C, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(C), a + b, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda()
def test_device_resident_with_host_arrays_warns():
    """NumPyCPU on a device-resident artifact: host->device copy + warning."""
    compiled, a, b, N = _compile_vec_add("cuda")
    c = np.zeros(N, dtype=np.float64)
    with pytest.warns(DoccPerformanceWarning, match="passed from host memory"):
        compiled(a, b, c, N)
    np.testing.assert_allclose(c, a + b, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda()
def test_device_resident_mixed_arrays_raises():
    """Mixing numpy and cupy arrays in a single call is rejected."""
    import cupy as cp

    compiled, a, b, N = _compile_vec_add("cuda")
    A = cp.asarray(a)  # device array
    C = cp.zeros(N, dtype=cp.float64)
    with pytest.raises(TypeError, match="Mixed array kinds"):
        compiled(A, b, C, N)  # b is a host numpy array -> mixed kinds


# --------------------------------------------------------------------------- #
# Host-resident artifact (prefix_sum) -- call-mode matrix
# --------------------------------------------------------------------------- #


@pytest.mark.cuda()
def test_host_resident_with_host_arrays_warns_promotion_failed():
    """A GPU-target host fallback warns that residency could not be enabled."""
    compiled, a, N = _compile_prefix_sum("cuda")
    work = a.copy()
    with pytest.warns(DoccPerformanceWarning, match="could not be enabled"):
        compiled(work, N)
    np.testing.assert_allclose(work, np.cumsum(a), rtol=1e-12, atol=1e-12)


@pytest.mark.cuda()
def test_host_resident_with_device_arrays_raises():
    """cupy arrays on a non-device-resident artifact are rejected."""
    import cupy as cp

    compiled, a, N = _compile_prefix_sum("cuda")
    work = cp.asarray(a)
    with pytest.raises(TypeError, match="not device-resident"):
        compiled(work, N)
