"""Test harness for individual pylulesh functions (NPBench-style).

Each LULESH function is checked by running the plain-NumPy version and the
docc-compiled (``native``, ``target="none"``) version on identical inputs and
asserting that all return values and (in-place) outputs match.

Two entry points:

- :func:`check_domain_kernel` for functions whose first argument is a ``Domain``
  (they read/mutate the struct's array members in place).
- :func:`check_flat_kernel` for helper functions that take only flat arrays.

Inputs are made deterministic and meaningful by (re)seeding selected ``Domain``
fields, since several fields are allocated with ``np.empty`` (garbage) and are
only populated by the full simulation pipeline.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

# Ensure both the pylulesh directory (for `util`, `domain`, `lulesh`) and the
# `python/` root (for `benchmarks.npbench.harness`) are importable, so the test
# files work both under pytest and when run directly as benchmark scripts
# (``python .../tests/test_foo.py``).
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYLULESH_DIR = os.path.dirname(_TESTS_DIR)
_PYTHON_ROOT = os.path.dirname(os.path.dirname(_PYLULESH_DIR))
for _p in (_PYLULESH_DIR, _PYTHON_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import util
from domain import Domain, RealT

from docc.python import native

from benchmarks.npbench.harness import SDFGVerification


def _get_loop_stats(compiled_program):
    """Return the SDFG loop/schedule node-count report for a compiled kernel."""
    sdfg = compiled_program.last_sdfg
    stats = sdfg.loop_report()
    assert stats is not None, "No stats found in SDFG."
    return stats


def _verify(verifier, compiled_program, name, target):
    if verifier is None:
        return
    stats = _get_loop_stats(compiled_program)
    verifier.verify(stats, name, target)


def make_domain(nx=3, seed=0, randomize=(), zeros=()):
    """Build a valid LULESH ``Domain`` for a ``nx``-cubed element mesh.

    ``randomize`` fields are filled with reproducible random values (floats get
    uniform [0, 1); integer fields get small non-negative ints). ``zeros``
    fields are set to 0. Connectivity/index fields should never be randomized.
    """
    col, row, plane, side = util.init_mesh_decomposition(1, 0)
    d = Domain(1, col, row, plane, nx, side, 1, 1, 1)

    rng = np.random.default_rng(seed)
    for name in randomize:
        arr = getattr(d, name)
        if arr.dtype.kind == "f":
            arr[:] = rng.random(arr.shape)
        else:
            arr[:] = rng.integers(0, 4, size=arr.shape).astype(arr.dtype)
    for name in zeros:
        getattr(d, name)[:] = 0
    return d


def _compare_returns(res_ref, res_docc, rtol, atol):
    if res_ref is None:
        return
    if isinstance(res_ref, tuple):
        assert isinstance(res_docc, (tuple, list)), "expected multiple return values"
        assert len(res_docc) == len(res_ref), "return arity mismatch"
        for i, (a, b) in enumerate(zip(res_docc, res_ref)):
            np.testing.assert_allclose(
                a, b, rtol=rtol, atol=atol, err_msg=f"return value [{i}] mismatch"
            )
    else:
        np.testing.assert_allclose(
            res_docc, res_ref, rtol=rtol, atol=atol, err_msg="return value mismatch"
        )


def _compare_inplace(args, args_ref, args_docc, rtol, atol):
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            np.testing.assert_allclose(
                args_docc[i],
                args_ref[i],
                rtol=rtol,
                atol=atol,
                err_msg=f"in-place argument [{i}] mismatch",
            )


def check_domain_kernel(
    kernel,
    target,
    *,
    nx=3,
    seed=0,
    randomize=(),
    zeros=(),
    extra_args=(),
    compare_fields,
    verifier=None,
    rtol=1e-9,
    atol=1e-9,
):
    """Run ``kernel(domain, *extra_args)`` under NumPy and docc and compare.

    ``compare_fields`` is the tuple of ``Domain`` array-member names the kernel
    is expected to write; only those are compared (avoiding uninitialized
    garbage fields). ``extra_args`` may be a tuple or a callable taking the
    per-run ``Domain`` and returning the argument tuple. ``verifier`` is an
    optional :class:`SDFGVerification` checking the compiled node counts.
    """
    base = make_domain(nx=nx, seed=seed, randomize=randomize, zeros=zeros)
    d_ref = copy.deepcopy(base)
    d_docc = copy.deepcopy(base)

    extra_ref = extra_args(d_ref) if callable(extra_args) else extra_args
    extra_docc = extra_args(d_docc) if callable(extra_args) else extra_args
    extra_ref = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in extra_ref)
    extra_docc = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in extra_docc)

    res_ref = kernel(d_ref, *extra_ref)
    compiled = native(kernel, target=target)
    res_docc = compiled(d_docc, *extra_docc)

    _compare_returns(res_ref, res_docc, rtol, atol)

    for f in compare_fields:
        np.testing.assert_allclose(
            getattr(d_docc, f),
            getattr(d_ref, f),
            rtol=rtol,
            atol=atol,
            err_msg=f"Domain field '{f}' mismatch",
        )

    _compare_inplace(extra_ref, extra_ref, extra_docc, rtol, atol)
    _verify(verifier, compiled, kernel.__name__, target)


def check_flat_kernel(kernel, target, args, *, verifier=None, rtol=1e-9, atol=1e-9):
    """Run ``kernel(*args)`` (flat-array signature) under NumPy and docc.

    Compares return values and any in-place-mutated ndarray arguments;
    ``verifier`` is an optional :class:`SDFGVerification` checking node counts.
    """
    args_ref = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)
    args_docc = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)

    res_ref = kernel(*args_ref)
    compiled = native(kernel, target=target)
    res_docc = compiled(*args_docc)

    _compare_returns(res_ref, res_docc, rtol, atol)
    _compare_inplace(args, args_ref, args_docc, rtol, atol)
    _verify(verifier, compiled, kernel.__name__, target)


# ---------------------------------------------------------------------------
# Benchmark drivers (used from the ``if __name__ == "__main__"`` guards).
# ---------------------------------------------------------------------------


def _bench_args(default_nx, args):
    if args is not None:
        return args
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=default_nx, help="elements per edge")
    parser.add_argument("--target", type=str, default="none")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--numpy", action="store_true", help="only run NumPy")
    parser.add_argument("--docc", action="store_true", help="only run docc")
    return parser.parse_args()


def _time(fn):
    start = time.time()
    fn()
    return time.time() - start


def run_domain_benchmark(
    kernel,
    name,
    *,
    default_nx=10,
    seed=0,
    randomize=(),
    zeros=(),
    extra_args=(),
    args=None,
):
    """Benchmark a Domain kernel (NumPy vs docc), scaling with ``--nx``."""
    args = _bench_args(default_nx, args)
    run_numpy = args.numpy or not args.docc
    run_docc = args.docc or not args.numpy

    base = make_domain(nx=args.nx, seed=seed, randomize=randomize, zeros=zeros)
    print(
        f"[{name}] nx={args.nx} numelem={base.numelem} numnode={base.numnode} "
        f"target={args.target}"
    )

    def _fresh():
        d = copy.deepcopy(base)
        extra = extra_args(d) if callable(extra_args) else extra_args
        extra = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in extra)
        return d, extra

    if run_numpy:
        for _ in range(args.n_runs):
            d, extra = _fresh()
            print(f"  numpy: {_time(lambda: kernel(d, *extra)):.6f} s")

    if run_docc:
        compiled = native(kernel, target=args.target)
        d, extra = _fresh()
        print(f"  docc (compile+run): {_time(lambda: compiled(d, *extra)):.6f} s")
        for _ in range(args.n_runs):
            d, extra = _fresh()
            print(f"  docc (cached): {_time(lambda: compiled(d, *extra)):.6f} s")


def run_flat_benchmark(kernel, name, make_args, *, default_nx=10, args=None):
    """Benchmark a flat-array kernel (NumPy vs docc).

    ``make_args(nx)`` returns the argument tuple for a given problem size.
    """
    args = _bench_args(default_nx, args)
    run_numpy = args.numpy or not args.docc
    run_docc = args.docc or not args.numpy

    base_args = make_args(args.nx)
    numelem = (
        base_args[0].shape[0] if base_args and hasattr(base_args[0], "shape") else 0
    )
    print(f"[{name}] nx={args.nx} numelem={numelem} target={args.target}")

    def _fresh():
        return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in base_args)

    if run_numpy:
        for _ in range(args.n_runs):
            a = _fresh()
            print(f"  numpy: {_time(lambda: kernel(*a)):.6f} s")

    if run_docc:
        compiled = native(kernel, target=args.target)
        a = _fresh()
        print(f"  docc (compile+run): {_time(lambda: compiled(*a)):.6f} s")
        for _ in range(args.n_runs):
            a = _fresh()
            print(f"  docc (cached): {_time(lambda: compiled(*a)):.6f} s")
