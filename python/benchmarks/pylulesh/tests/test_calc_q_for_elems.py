"""Test + benchmark for LULESH ``calc_q_for_elems`` (monotonic artificial
viscosity).

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_q_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd", "vnew")
_COMPARE = (
    "delv_xi",
    "delv_eta",
    "delv_zeta",
    "delx_xi",
    "delx_eta",
    "delx_zeta",
    "ql",
    "qq",
    "q",
)


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_q_for_elems(target):
    check_domain_kernel(
        lulesh.calc_q_for_elems,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_q_for_elems,
        "calc_q_for_elems",
        randomize=_RANDOMIZE,
    )
