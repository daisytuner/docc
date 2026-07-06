"""Test + benchmark for LULESH ``lagrange_nodal`` -- the full nodal update.

``lagrange_nodal`` chains the entire nodal half of a LULESH timestep:
force calculation (stress integration + hourglass control), acceleration,
symmetry boundary conditions, velocity and position updates. It is the
largest single LULESH kernel and a good end-to-end benchmark.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_lagrange_nodal.py --nx 8
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

# Seed the physics fields that drive the force computation (stress p/q, sound
# speed ss for hourglass, and the nodal velocities).
_RANDOMIZE = ("p", "q", "ss", "xd", "yd", "zd")
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
)


@pytest.mark.skip(reason="Slow in CI; run manually for benchmarking")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_lagrange_nodal(target):
    check_domain_kernel(
        lulesh.lagrange_nodal,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.lagrange_nodal,
        "lagrange_nodal",
        default_nx=6,
        randomize=_RANDOMIZE,
    )
