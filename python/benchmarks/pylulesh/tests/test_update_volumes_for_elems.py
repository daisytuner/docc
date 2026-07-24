"""Test + benchmark for LULESH ``update_volumes_for_elems``.

Applies the relative-volume cutoff: ``v = where(|vnew - 1| < v_cut, 1, vnew)``.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_update_volumes_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("vnew",)
_COMPARE = ("v",)
_EXTRA = (1e-8,)  # v_cut


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_update_volumes_for_elems(target):
    check_domain_kernel(
        lulesh.update_volumes_for_elems,
        target,
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.update_volumes_for_elems,
        "update_volumes_for_elems",
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
    )
