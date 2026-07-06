"""Test + benchmark for LULESH ``lagrange_elements`` -- the full element half of
a timestep (kinematics, artificial viscosity, EOS/material update, volumes).

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_lagrange_elements.py --nx 12
"""

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd", "e")
_COMPARE = (
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
def test_lagrange_elements(target):
    check_domain_kernel(
        lulesh.lagrange_elements,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.lagrange_elements,
        "lagrange_elements",
        default_nx=8,
        randomize=_RANDOMIZE,
    )
