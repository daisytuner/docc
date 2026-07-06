"""Test + benchmark for LULESH ``calc_time_constraints_for_elems``.

Exercises scalar struct-member writes that round-trip back to the Domain
(``domain.dtcourant`` / ``domain.dthydro``), the variable-length region slice
``reg_elem_list[r, :reg_elem_size[r]]``, and the min-reduction guard functions
inlined with conditional (early) returns.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_time_constraints_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("ss", "vdov", "arealg")
_COMPARE = ("dtcourant", "dthydro")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_time_constraints_for_elems(target):
    check_domain_kernel(
        lulesh.calc_time_constraints_for_elems,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_time_constraints_for_elems,
        "calc_time_constraints_for_elems",
        randomize=_RANDOMIZE,
    )
