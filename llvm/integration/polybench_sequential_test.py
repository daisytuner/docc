import subprocess
import numpy as np
import pandas as pd

import re
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, SDFGVerification


def verify(reference_file: Path, test_file: Path, dtype, max_ulps=None):
    cmd = [reference_file]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    data = stderr.splitlines()
    header = data.pop(0)
    assert header == "==BEGIN DUMP_ARRAYS=="

    dtype_ = dtype
    reference_arrays = {}
    current_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if line.startswith("begin"):
            assert current_array is None
            current_array = line[line.find(":") + 1 :].strip()
            reference_arrays[current_array] = []
        elif line.startswith("end"):
            assert (
                current_array is not None
                and current_array == line[line.find(":") + 1 :].strip()
            )
            reference_arrays[current_array] = np.hstack(reference_arrays[current_array])
            current_array = None
        else:
            assert current_array is not None
            row = np.fromstring(line, dtype=dtype_, sep=" ")
            reference_arrays[current_array].append(row)

    cmd = [test_file]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    data = stderr.splitlines()
    header = data.pop(0)
    assert header == "==BEGIN DUMP_ARRAYS=="

    test_arrays = {}
    current_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if line.startswith("begin"):
            assert current_array is None
            current_array = line[line.find(":") + 1 :].strip()
            test_arrays[current_array] = []
        elif line.startswith("end"):
            assert (
                current_array is not None
                and current_array == line[line.find(":") + 1 :].strip()
            )
            test_arrays[current_array] = np.hstack(test_arrays[current_array])
            current_array = None
        else:
            assert current_array is not None
            row = np.fromstring(line, dtype=dtype, sep=" ")
            test_arrays[current_array].append(row)

    for array in reference_arrays:
        if max_ulps:
            tol = max_ulps * np.abs(np.spacing(reference_arrays[array]))
            assert np.all(np.abs(test_arrays[array] - reference_arrays[array]) <= tol)
        else:
            assert np.array_equal(
                reference_arrays[array], test_arrays[array], equal_nan=True
            )


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_correlation(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "datamining" / "correlation"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "SEQUENTIAL": 25, "MAP": 17, "FOR": 4},
    )
    test_case = benchmark_path / "correlation.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_covariance(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 13, "SEQUENTIAL": 20, "FOR": 4, "REDUCE": 3},
    )
    test_case = benchmark_path / "covariance.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_gemm(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "gemm"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "REDUCE": 3,
            "SEQUENTIAL": 13,
            "MAP": 8,
            "FOR": 2,
            "GEMM": 1,
        },
    )
    test_case = benchmark_path / "gemm.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e2,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_gemver(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "gemver"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 7, "SEQUENTIAL": 13, "FOR": 2},
    )
    test_case = benchmark_path / "gemver.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e2,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_gesummv(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "gesummv"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "SEQUENTIAL": 10, "MAP": 5, "FOR": 1},
    )
    test_case = benchmark_path / "gesummv.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_symm(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "symm"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 7, "SEQUENTIAL": 14, "FOR": 3}
    )
    test_case = benchmark_path / "symm.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_syr2k(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "syr2k"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "SEQUENTIAL": 14, "MAP": 8, "FOR": 2},
    )
    test_case = benchmark_path / "syr2k.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_syrk(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "syrk"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 8, "SEQUENTIAL": 14, "FOR": 2},
    )
    test_case = benchmark_path / "syrk.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_trmm(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "blas"
        / "trmm"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 9, "SEQUENTIAL": 15, "FOR": 3, "REDUCE": 3},
    )
    test_case = benchmark_path / "trmm.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_2mm(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "2mm"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "MAP": 12,
            "SEQUENTIAL": 17,
            "FOR": 2,
            "GEMM": 2,
            "REDUCE": 3,
        },
    )
    test_case = benchmark_path / "2mm.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e2,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_3mm(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "3mm"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "MAP": 14,
            "SEQUENTIAL": 19,
            "FOR": 2,
            "GEMM": 3,
            "REDUCE": 3,
        },
    )
    test_case = benchmark_path / "3mm.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e2,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_atax(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "atax"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 7, "SEQUENTIAL": 13, "FOR": 2},
    )
    test_case = benchmark_path / "atax.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_bicg(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "bicg"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 6, "SEQUENTIAL": 12, "FOR": 3, "REDUCE": 3},
    )
    test_case = benchmark_path / "bicg.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_doitgen(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "doitgen"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 9, "REDUCE": 3, "SEQUENTIAL": 23, "FOR": 9, "MAP": 11},
    )
    test_case = benchmark_path / "doitgen.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_mvt(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "kernels"
        / "mvt"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 4, "SEQUENTIAL": 11, "FOR": 3},
    )
    test_case = benchmark_path / "mvt.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_cholesky(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "cholesky"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 6, "SEQUENTIAL": 21, "MAP": 11, "FOR": 4},
    )
    test_case = benchmark_path / "cholesky.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_durbin(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "durbin"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "MAP": 3, "SEQUENTIAL": 9, "FOR": 2},
    )
    test_case = benchmark_path / "durbin.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_gramschmidt(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "gramschmidt"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 4, "SEQUENTIAL": 19, "MAP": 9, "FOR": 6},
    )
    test_case = benchmark_path / "gramschmidt.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_lu(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "lu"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 6, "SEQUENTIAL": 22, "MAP": 12, "FOR": 4},
    )
    test_case = benchmark_path / "lu.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_ludcmp(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "ludcmp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 8, "SEQUENTIAL": 26, "MAP": 13, "FOR": 5},
    )
    test_case = benchmark_path / "ludcmp.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_trisolv(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent
        / "tests"
        / "polybench"
        / "linear-algebra"
        / "solvers"
        / "trisolv"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 4, "SEQUENTIAL": 9, "FOR": 2, "REDUCE": 3},
    )
    test_case = benchmark_path / "trisolv.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_deriche(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "deriche"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 10, "SEQUENTIAL": 19, "FOR": 6, "REDUCE": 3},
    )
    test_case = benchmark_path / "deriche.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


def test_floyd_warshall(compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "floyd-warshall"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 2, "SEQUENTIAL": 10, "FOR": 5, "REDUCE": 3},
    )
    test_case = benchmark_path / "floyd-warshall.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            "-DDATA_TYPE_IS_INT",
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


def test_nussinov(compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "nussinov"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "REDUCE": 3,
            "SEQUENTIAL": 10,
            "MAP": 3,
            "FOR": 4,
            "WHILE": 1,
        },
    )
    test_case = benchmark_path / "nussinov.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            "-DDATA_TYPE_IS_INT",
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_adi(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = Path(__file__).parent / "tests" / "polybench" / "stencils" / "adi"

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 10, "SEQUENTIAL": 20, "FOR": 7, "REDUCE": 3},
    )
    test_case = benchmark_path / "adi.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_fdtd_2d(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "stencils" / "fdtd-2d"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 3, "SEQUENTIAL": 20, "MAP": 10, "FOR": 7},
    )
    test_case = benchmark_path / "fdtd-2d.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_heat_3d(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "stencils" / "heat-3d"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "REDUCE": 3, "SEQUENTIAL": 16, "MAP": 9, "FOR": 4},
    )
    test_case = benchmark_path / "heat-3d.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_jacobi_1d(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-1d"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 3, "SEQUENTIAL": 8, "FOR": 2, "REDUCE": 3},
    )
    test_case = benchmark_path / "jacobi-1d.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_jacobi_2d(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-2d"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 6, "SEQUENTIAL": 12, "FOR": 3, "REDUCE": 3},
    )
    test_case = benchmark_path / "jacobi-2d.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        pytest.param("-DDATA_TYPE_IS_FLOAT"),
    ],
)
def test_seidel_2d(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "stencils" / "seidel-2d"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "MAP": 2, "SEQUENTIAL": 10, "FOR": 5, "REDUCE": 3},
    )
    test_case = benchmark_path / "seidel-2d.c"
    runner = TestRunner(
        "Polybench",
        test_case,
        "docc",
        compiler,
        [
            "-g",
            "-O3",
            "-DPOLYBENCH_DUMP_ARRAYS",
            "-D" + size,
            datatype,
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "polybench" / "utilities").absolute()
            ),
            "-lm",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)
