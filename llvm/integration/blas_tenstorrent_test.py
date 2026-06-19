import subprocess
import pytest
import numpy as np

from functools import partial
from pathlib import Path

from test_runner import TestRunner, SDFGVerification


def verify(
    reference_file: Path,
    test_file: Path,
    dtype,
    rtol: float = 1e-05,
    atol: float = 1e-08,
):
    cmd = [reference_file]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    data = stdout.splitlines()
    # Skip lines until we find the header
    while data:
        header = data.pop(0)
        if header == "==BEGIN DUMP_ARRAYS==":
            break
    else:
        raise AssertionError(
            "Header '==BEGIN DUMP_ARRAYS==' not found in reference output"
        )

    dtype_ = dtype
    reference_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if reference_array is None:
            reference_array = []

        row = np.fromstring(line, dtype=dtype_, sep=" ")
        reference_array.append(row)

    cmd = [test_file]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("Crash: STDERR ==============")
        print(stderr)
        print("Crash: STDOUT ==============")
        print(stdout)
    assert process.returncode == 0

    data = stdout.splitlines()
    # Skip lines until we find the header
    while data:
        header = data.pop(0)
        if header == "==BEGIN DUMP_ARRAYS==":
            break
    else:
        raise AssertionError(
            "Header '==BEGIN DUMP_ARRAYS==' not found in reference output"
        )

    dtype_ = dtype
    test_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if test_array is None:
            test_array = []

        row = np.fromstring(line, dtype=dtype_, sep=" ")
        test_array.append(row)

    assert np.allclose(reference_array, test_array, rtol=rtol, atol=atol)


@pytest.mark.skip(reason="docc-expand will be deprecated")
@pytest.mark.parametrize(
    "precision, expand, rtol",
    [
        pytest.param("s", "tenstorrent", 1e-03),
    ],
)
def test_gemm(precision: str, expand: str, rtol: float):
    benchmark_path = Path(__file__).parent / "tests" / "blas"

    verifier_no_expand = SDFGVerification(
        verification={
            "sdfgs": 1,
            "GEMM": 1,
            "FOR": 5,
            "SEQUENTIAL": 2,
            "MAP": 2,
        },
    )
    verifier_expand = SDFGVerification(
        verification={
            "sdfgs": 1,
            "FOR": 8,
            "MAP": 5,
            "SEQUENTIAL": 5,
        },
    )
    test_case = benchmark_path / "gemm.c"
    runner = TestRunner(
        "BLAS",
        test_case,
        "docc",
        "clang-19",
        [
            "-g",
            f"-DDATA_TYPE_IS_{precision}",
            "-O3",
            "-lblas",
        ],
        "none",
        [],
        partial(
            verify, dtype=np.float64 if precision == "d" else np.float32, rtol=rtol
        ),
        sdfg_verification=verifier_expand if expand == "all" else verifier_no_expand,
        docc_flags=["-mllvm", "-docc-expand=" + expand],
    )
    return runner.run()

@pytest.mark.skip(reason="docc-expand will be deprecated")
@pytest.mark.parametrize(
    "precision, expand, rtol",
    [
        pytest.param("s", "tenstorrent", 1e-03),
    ],
)
def test_dot(precision: str, expand: str, rtol: float):
    benchmark_path = Path(__file__).parent / "tests" / "blas"

    verifier_no_expand = SDFGVerification(
        verification={
            "sdfgs": 1,
            "DOT": 1,
            "FOR": 5,
            "SEQUENTIAL": 1,
            "MAP": 1,
        },
    )
    verifier_expand = SDFGVerification(
        verification={
            "sdfgs": 1,
            "FOR": 4,
            "MAP": 3,
            "TTOffloading": 4,
            "SEQUENTIAL": 3,
        },
    )
    test_case = benchmark_path / "dot.c"
    runner = TestRunner(
        "BLAS",
        test_case,
        "docc",
        "clang-19",
        [
            "-g",
            f"-DDATA_TYPE_IS_{precision}",
            "-O3",
            "-lblas",
        ],
        "sequential",
        [],
        partial(
            verify, dtype=np.float64 if precision == "d" else np.float32, rtol=rtol
        ),
        sdfg_verification=verifier_expand if expand == "tenstorrent" else verifier_no_expand,
        docc_flags=["-mllvm", "-docc-expand=" + expand],
    )
    return runner.run()