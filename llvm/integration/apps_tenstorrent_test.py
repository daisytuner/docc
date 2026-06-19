import subprocess
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, SDFGVerification


def evaluate_hpccg(reference_file: Path, test_file: Path, args) -> float:
    cmd = [reference_file] + args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    data = stdout.splitlines()
    header = data.pop(0)
    while not header.startswith("Final residual"):
        header = data.pop(0).strip()

    ref_final_residual = np.array([float(header.split(":")[1].strip())])

    cmd = [test_file] + args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    data = stdout.splitlines()
    header = data.pop(0)
    while not header.startswith("Final residual"):
        header = data.pop(0).strip()

    test_final_residual = np.array([float(header.split(":")[1].strip())])
    print("###", test_final_residual, ref_final_residual)
    assert np.abs(test_final_residual - ref_final_residual) <= 1e-17

@pytest.mark.skip("Dot expansion regression")
def test_HPCCG():
    test_case = Path(__file__).parent / "tests" / "apps" / "HPCCG" / "main.cpp"
    verifier = SDFGVerification(
        verification={"FOR": 22, "MAP": 9, "WHILE": 6, 'SEQUENTIAL': 3, "TTOffloading": 18, "DOT": 1},
    )
    runner = TestRunner(
        "Apps",
        test_case,
        "docc-cpp",
        "clang++-19",
        ["-O3", "-g", "-lm"],
        "tenstorrent",
        [
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "compute_residual.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "ddot.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "generate_matrix.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "HPC_Sparse_Matrix.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "HPC_sparsemv.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "HPCCG.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "mytimer.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "waxpby.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "YAML_Doc.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "YAML_Element.cpp",
        ],
        partial(
            evaluate_hpccg,
            args=["50", "50", "50"],
        ),
        sdfg_verification=verifier,
        docc_flags=["-docc-einsum", "-docc-no-offloading-transfer-opt"],
    )
    runner.run()
