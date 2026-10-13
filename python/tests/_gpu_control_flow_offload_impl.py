"""
Backend-agnostic integration tests for offloading root-level control flow.

The GPU offload schedulers wrap runs of non-loop root children (blocks, if/else,
while loops, sequences) into a single-iteration map and offload them, so host
code between, before and after offloaded loops runs on the device as well. These
tests compile small programs through the full pipeline and compare every array
argument (inputs and outputs) and the return value against plain Python
execution of the very same function.

The scenarios vary how outputs are produced: written only by root code, only by
loops, by both, read back by root code after a loop, used as intermediates
between loops, written conditionally or inside a data-dependent while loop,
returned as a scalar, or routed through a local temporary. Device-resident
artifacts are additionally called with device arrays (cupy), which is the path
taken by framework integrations.

The thin ``cuda`` / ``rocm`` suites bind a target via :func:`register`.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pytest

from docc.compiler import DoccPerformanceWarning
from docc.python import native


# ---------------------------------------------------------------------------
# Scenarios. Each function is executed twice: compiled by docc and as plain
# Python on NumPy copies (the reference). Root code only indexes 0 and N - 1 so
# every scenario is valid for N == 1.
# ---------------------------------------------------------------------------
def cf_root_only(A, B, N):
    B[0] = A[0] + A[N - 1]
    B[N - 1] = A[0] * A[N - 1]


def cf_root_init_then_reduce(x, y, out, N):
    out[0] = 0.0
    for i in range(N):
        out[0] += x[i] * y[i]


def cf_loop_then_root_update(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]
    C[0] = C[0] * 2.0
    C[N - 1] = C[N - 1] - A[0]


def cf_root_between_loops(A, B, C, N):
    for i in range(N):
        B[i] = A[i] * 2.0
    B[0] = -1.0
    for i in range(N):
        C[i] = B[i] + 1.0


def cf_root_reads_loop_result(A, B, out, N):
    for i in range(N):
        B[i] = A[i] * A[i]
    out[0] = B[0] + B[N - 1]


def cf_mixed_outputs(A, X, Y, Z, N):
    X[0] = A[N - 1]
    for i in range(N):
        Y[i] = A[i] + 1.0
        Z[i] = A[i] * 3.0
    Z[0] = Z[0] + X[0]


def cf_in_place_root(A, B, N):
    A[0] = A[0] + A[N - 1]
    for i in range(N):
        B[i] = A[i] * 0.5


def cf_conditional_root(A, B, out, N):
    for i in range(N):
        B[i] = A[i] - 0.5
    if B[0] > 0.0:
        out[0] = B[0]
        out[1] = 1.0
    else:
        out[0] = -B[0]
        out[1] = -1.0


def cf_root_while_break(A, B, N):
    k = 0
    while k < N:
        B[k] = A[k] + 1.0
        if B[k] > 10.0:
            break
        k = k + 1


def cf_scalar_return(A, N) -> float:
    for i in range(N):
        A[i] = A[i] * 2.0
    return A[0] + A[N - 1]


def cf_local_temporary(A, out, N):
    tmp = np.zeros(N)
    for i in range(N):
        tmp[i] = A[i] * 3.0
    out[0] = tmp[0] + tmp[N - 1]


def cf_two_dimensional(A, C, N):
    for i in range(N):
        for j in range(N):
            C[i, j] = A[i, j] + A[j, i]
    C[0, N - 1] = A[N - 1, 0]
    C[N - 1, 0] = C[0, 0] * 2.0


@dataclass(frozen=True)
class Scenario:
    fn: Callable
    make_args: Callable  # (n, dtype, rng) -> list of arguments
    # Expected device residency of the compiled artifact; None leaves it unchecked.
    resident: Optional[bool]
    min_n: int = 1
    skip: Optional[str] = None

    @property
    def id(self):
        return self.fn.__name__[len("cf_") :]

    def param(self):
        marks = [pytest.mark.skip(reason=self.skip)] if self.skip else []
        return pytest.param(self, id=self.id, marks=marks)


def _vec(rng, n, dtype):
    return rng.random(n).astype(dtype)


def _zeros(n, dtype):
    return np.zeros(n, dtype=dtype)


def _while_args(n, dtype, rng):
    a = _vec(rng, n, dtype)
    # Forces the data-dependent break in the middle of the array.
    a[n // 2] = 20.0
    return [a, _zeros(n, dtype), n]


SCENARIOS = [
    Scenario(
        cf_root_only,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(n, dt), n],
        resident=True,
    ),
    # The scalar reduction loop itself is not offloaded.
    Scenario(
        cf_root_init_then_reduce,
        lambda n, dt, rng: [_vec(rng, n, dt), _vec(rng, n, dt), _zeros(1, dt), n],
        resident=None,
    ),
    Scenario(
        cf_loop_then_root_update,
        lambda n, dt, rng: [_vec(rng, n, dt), _vec(rng, n, dt), _zeros(n, dt), n],
        resident=True,
    ),
    Scenario(
        cf_root_between_loops,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(n, dt), _zeros(n, dt), n],
        resident=True,
    ),
    Scenario(
        cf_root_reads_loop_result,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(n, dt), _zeros(1, dt), n],
        resident=True,
    ),
    Scenario(
        cf_mixed_outputs,
        lambda n, dt, rng: [
            _vec(rng, n, dt),
            _zeros(1, dt),
            _zeros(n, dt),
            _zeros(n, dt),
            n,
        ],
        resident=True,
    ),
    Scenario(
        cf_in_place_root,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(n, dt), n],
        resident=True,
    ),
    Scenario(
        cf_conditional_root,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(n, dt), _zeros(2, dt), n],
        resident=True,
    ),
    # `k = 0` is a host assignment, so the while loop writes a host-visible scalar.
    Scenario(cf_root_while_break, _while_args, resident=None),
    # The scalar return value has to be produced on the host.
    Scenario(
        cf_scalar_return,
        lambda n, dt, rng: [_vec(rng, n, dt), n],
        resident=None,
        # Faults on ROCm and poisons the context for all following tests.
        skip="device-resident calls pass scalar outputs as host buffers",
    ),
    # The temporary's allocation stays on the host.
    Scenario(
        cf_local_temporary,
        lambda n, dt, rng: [_vec(rng, n, dt), _zeros(1, dt), n],
        resident=None,
    ),
    # Root accesses through the stride symbol have no statically known argument size.
    Scenario(
        cf_two_dimensional,
        lambda n, dt, rng: [
            rng.random((n, n)).astype(dt),
            np.zeros((n, n), dtype=dt),
            n,
        ],
        resident=None,
        # Size-1 dimensions are specialized but the artifact hash ignores shapes, so an
        # N == 1 build would be reused for larger N.
        min_n=2,
    ),
]

SIZES = [1, 37, 1024]
DTYPES = [np.float64, np.float32]


def _copy_args(args):
    return [a.copy() if isinstance(a, np.ndarray) else a for a in args]


def _assert_matches(actual_args, expected_args, actual_ret, expected_ret, dtype):
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    for index, (actual, expected) in enumerate(zip(actual_args, expected_args)):
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=rtol, err_msg=f"argument {index}"
            )
    if expected_ret is not None:
        np.testing.assert_allclose(actual_ret, expected_ret, rtol=rtol, atol=rtol)


def _call(compiled, args):
    with warnings.catch_warnings():
        # Host arrays on a device-resident artifact (or a host fallback) warn by design.
        warnings.simplefilter("ignore", DoccPerformanceWarning)
        return compiled(*args)


# ===========================================================================
# Suite registration: bind the parametrized test to a concrete target and
# inject it into the calling (thin) suite module's globals. The thin suite's
# module-level ``pytestmark`` (cuda / rocm) then applies to the injected test.
# ===========================================================================
def register(namespace, target, device_module):
    """Define the control-flow offload test, bound to ``target``, into ``namespace``.

    ``device_module`` names the module providing device arrays (``cupy``) and is
    imported lazily, so collection works without it.
    """

    @pytest.mark.parametrize("dtype", DTYPES, ids=["f64", "f32"])
    @pytest.mark.parametrize("n", SIZES, ids=[f"n{n}" for n in SIZES])
    @pytest.mark.parametrize("scenario", [s.param() for s in SCENARIOS])
    def test_control_flow_offload(scenario, n, dtype):
        if n < scenario.min_n:
            pytest.skip(f"scenario requires N >= {scenario.min_n}")
        rng = np.random.default_rng(n)
        args = scenario.make_args(n, dtype, rng)

        expected_args = _copy_args(args)
        expected_ret = scenario.fn(*expected_args)

        program = native(scenario.fn, target=target, category="server")
        compiled = program.compile(*_copy_args(args))
        if scenario.resident is not None:
            assert compiled.device_resident is scenario.resident

        host_args = _copy_args(args)
        host_ret = _call(compiled, host_args)
        _assert_matches(host_args, expected_args, host_ret, expected_ret, dtype)

        if not compiled.device_resident:
            return

        xp = pytest.importorskip(device_module)
        device_args = [xp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
        device_ret = _call(compiled, device_args)
        xp.cuda.runtime.deviceSynchronize()
        _assert_matches(
            [xp.asnumpy(a) if isinstance(a, xp.ndarray) else a for a in device_args],
            expected_args,
            device_ret,
            expected_ret,
            dtype,
        )

    namespace.update(test_control_flow_offload=test_control_flow_offload)
