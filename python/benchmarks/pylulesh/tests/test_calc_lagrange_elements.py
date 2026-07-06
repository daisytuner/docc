"""Test + benchmark for LULESH ``calc_lagrange_elements``.

Computes the element strain rates (kinematics) and the deviatoric rate of
deformation, and updates ``vdov``.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_lagrange_elements.py --nx 20
"""

import sys
import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd")
_COMPARE = ("vnew", "delv", "arealg", "dxx", "dyy", "dzz", "vdov")


@pytest.mark.skip(reason="Slow in CI; run manually for benchmarking")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_lagrange_elements(target):
    check_domain_kernel(
        lulesh.calc_lagrange_elements,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_lagrange_elements,
        "calc_lagrange_elements",
        randomize=_RANDOMIZE,
    )
