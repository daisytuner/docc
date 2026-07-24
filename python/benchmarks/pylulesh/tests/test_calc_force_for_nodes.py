"""Test + benchmark for LULESH ``calc_force_for_nodes``.

Computes the full nodal force field: stress integration plus Flanagan-Belytschko
anti-hourglass control, scattered into ``fx/fy/fz``.
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("p", "q", "ss", "xd", "yd", "zd")
_COMPARE = ("fx", "fy", "fz")


@pytest.mark.skip(reason="Slow in CI; run manually for benchmarking")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_force_for_nodes(target):
    check_domain_kernel(
        lulesh.calc_force_for_nodes,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_force_for_nodes,
        "calc_force_for_nodes",
        default_nx=6,
        randomize=_RANDOMIZE,
    )
