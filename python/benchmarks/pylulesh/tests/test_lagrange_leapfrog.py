"""Test + benchmark for LULESH ``lagrange_leapfrog`` -- one full timestep
(nodal half + element half).

This is the end-to-end "run LULESH fully" kernel.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_lagrange_leapfrog.py --nx 10
"""

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("p", "q", "ss", "xd", "yd", "zd", "e")
_COMPARE = (
    "fx",
    "fy",
    "fz",
    "xdd",
    "ydd",
    "zdd",
    "xd",
    "yd",
    "zd",
    "x",
    "y",
    "z",
    "vnew",
    "delv",
    "arealg",
    "dxx",
    "dyy",
    "dzz",
    "vdov",
    "ql",
    "qq",
    "p",
    "e",
    "q",
    "ss",
    "v",
)


@pytest.mark.skip(reason="Slow in CI; run manually for benchmarking")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_lagrange_leapfrog(target):
    check_domain_kernel(
        lulesh.lagrange_leapfrog,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.lagrange_leapfrog,
        "lagrange_leapfrog",
        default_nx=6,
        randomize=_RANDOMIZE,
    )
