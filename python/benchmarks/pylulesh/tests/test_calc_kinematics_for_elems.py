"""Test + benchmark for LULESH ``calc_kinematics_for_elems``.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_kinematics_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

# Nodal velocities drive the velocity-gradient (strain-rate) computation.
_RANDOMIZE = ("xd", "yd", "zd")
_COMPARE = ("vnew", "delv", "arealg", "dxx", "dyy", "dzz")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_kinematics_for_elems(target):
    check_domain_kernel(
        lulesh.calc_kinematics_for_elems,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_kinematics_for_elems,
        "calc_kinematics_for_elems",
        randomize=_RANDOMIZE,
    )
