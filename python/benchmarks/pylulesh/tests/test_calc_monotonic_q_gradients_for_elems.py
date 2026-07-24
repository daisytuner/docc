"""Test + benchmark for LULESH ``calc_monotonic_q_gradients_for_elems``.

Computes the per-element velocity/coordinate gradients across the element faces
(delv_xi/eta/zeta, delx_xi/eta/zeta).

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_monotonic_q_gradients_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd", "vnew")
_COMPARE = ("delv_xi", "delv_eta", "delv_zeta", "delx_xi", "delx_eta", "delx_zeta")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_monotonic_q_gradients_for_elems(target):
    check_domain_kernel(
        lulesh.calc_monotonic_q_gradients_for_elems,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_monotonic_q_gradients_for_elems,
        "calc_monotonic_q_gradients_for_elems",
        randomize=_RANDOMIZE,
    )
