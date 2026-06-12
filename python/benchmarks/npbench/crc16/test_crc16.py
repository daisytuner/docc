# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 1600},
    "M": {"N": 16000},
    "L": {"N": 160000},
    "paper": {"N": 1000000},
}


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N,), dtype=np.uint8)

    poly = 0x8408
    return data, poly


# Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
def kernel(data, poly):
    """
    CRC-16-CCITT Algorithm
    """
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


@pytest.mark.parametrize(
    "target",
    ["none", "sequential", "openmp", "cuda", "rocm"],
)
def test_crc16(target):
    verifier = None
    if target == "none":
        verifier = SDFGVerification(verification={"FOR": 2})
    elif target == "sequential":
        verifier = SDFGVerification(verification={"FOR": 2})
    elif target == "openmp":
        verifier = SDFGVerification(verification={"FOR": 2})
    elif target == "cuda":
        verifier = SDFGVerification(verification={"FOR": 2})
    elif target == "rocm":
        verifier = SDFGVerification(verification={"FOR": 2})
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "crc16")
