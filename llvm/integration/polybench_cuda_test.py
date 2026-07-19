import subprocess
import os
import numpy as np
import pandas as pd

import re
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, SDFGVerification


def verify(reference_file: Path, test_file: Path, dtype, max_ulps):
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
    omp_env = os.environ.copy()
    omp_env["OMP_NUM_THREADS"] = "10"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=omp_env,
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
        tol = max_ulps * np.abs(np.spacing(reference_arrays[array]))
        assert np.all(np.abs(test_arrays[array] - reference_arrays[array]) <= tol)


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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "SEQUENTIAL": 13,
            "CUDAOffloading": 26,
            "MAP": 17,
            "CUDA": 12,
            "FOR": 4,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 18,
            "MAP": 13,
            "CUDA": 9,
            "FOR": 4,
            "SEQUENTIAL": 11,
            "REDUCE": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
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
            "FOR": 6,
            "SEQUENTIAL": 6,
            "CUDAOffloading": 14,
            "MAP": 10,
            "CUDA": 10,
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
        ],
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=[],
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 38,
            "MAP": 6,
            "CUDA": 5,
            "FOR": 2,
            "SEQUENTIAL": 7,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 24,
            "MAP": 4,
            "CUDA": 4,
            "SEQUENTIAL": 5,
            "FOR": 1,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 8,
            "MAP": 7,
            "CUDA": 4,
            "SEQUENTIAL": 10,
            "FOR": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 14,
            "MAP": 8,
            "CUDA": 6,
            "SEQUENTIAL": 8,
            "FOR": 2,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 10,
            "MAP": 8,
            "CUDA": 6,
            "SEQUENTIAL": 8,
            "FOR": 2,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE",
            marks=pytest.mark.xfail(reason="Verifier changed & Program crashes"),
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT",
            marks=pytest.mark.xfail(reason="Verifier changed & Program crashes"),
        ),
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
        verification={
            "sdfgs": 8,
            "FOR": 14,
            "MAP": 7,
            "SEQUENTIAL": 2,
            "CUDA": 5,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        # sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
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
            "FOR": 25,
            "barrier_local": 8,
            "CUDAOffloading": 13,
            "MAP": 16,
            "CUDA": 16,
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
        ],
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=[],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
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
            "FOR": 31,
            "barrier_local": 12,
            "CUDAOffloading": 17,
            "MAP": 20,
            "CUDA": 20,
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
        ],
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=[],
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 14,
            "MAP": 7,
            "CUDA": 6,
            "SEQUENTIAL": 7,
            "FOR": 2,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE",
            marks=pytest.mark.xfail(reason="Verifier changed & Output incorrect"),
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT",
            marks=pytest.mark.xfail(reason="Verifier changed & Output incorrect"),
        ),
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
        verification={
            "sdfgs": 8,
            "FOR": 12,
            "MAP": 5,
            "CUDA": 5,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 9,
            "REDUCE": 3,
            "SEQUENTIAL": 18,
            "CUDA": 4,
            "FOR": 9,
            "CUDAOffloading": 4,
            "MAP": 10,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 4,
            "CUDAOffloading": 16,
            "MAP": 3,
            "CUDA": 2,
            "SEQUENTIAL": 8,
            "FOR": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 6,
            "SEQUENTIAL": 12,
            "CUDAOffloading": 16,
            "MAP": 11,
            "CUDA": 9,
            "FOR": 4,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        # pytest.param("-DDATA_TYPE_IS_DOUBLE"),
        # pytest.param("-DDATA_TYPE_IS_FLOAT"),
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
        verification={
            "sdfgs": 8,
            "FOR": 9,
            "MAP": 3,
            "SEQUENTIAL": 0,
            "CUDA": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.skip(reason="Wrong results, needs investigation")
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
        verification={
            "sdfgs": 8,
            "FOR": 19,
            "MAP": 9,
            "SEQUENTIAL": 3,
            "CUDA": 6,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 6,
            "SEQUENTIAL": 13,
            "CUDAOffloading": 16,
            "MAP": 12,
            "CUDA": 9,
            "FOR": 4,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 8,
            "Call": 18,
            "SEQUENTIAL": 16,
            "CUDAOffloading": 22,
            "MAP": 13,
            "CUDA": 10,
            "FOR": 5,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE",
            marks=pytest.mark.xfail(reason="Verifier changed & Output incorrect"),
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT",
            marks=pytest.mark.xfail(reason="Verifier changed & Output incorrect"),
        ),
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
        verification={
            "sdfgs": 8,
            "FOR": 8,
            "MAP": 2,
            "SEQUENTIAL": 1,
            "CUDA": 1,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
    ],
)
def test_deriche(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "deriche"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 22,
            "MAP": 10,
            "FOR": 19,
            "CUDA": 10,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 2,
            "MAP": 2,
            "CUDA": 2,
            "FOR": 5,
            "SEQUENTIAL": 8,
            "REDUCE": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64, max_ulps=1e6),
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
            "CUDAOffloading": 4,
            "MAP": 3,
            "CUDA": 3,
            "FOR": 4,
            "SEQUENTIAL": 7,
            "REDUCE": 3,
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64, max_ulps=1e6),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)


@pytest.mark.parametrize(
    "datatype",
    [
        pytest.param(
            "-DDATA_TYPE_IS_DOUBLE", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
        pytest.param(
            "-DDATA_TYPE_IS_FLOAT", marks=pytest.mark.xfail(reason="Verifier changed")
        ),
    ],
)
def test_adi(datatype, compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = Path(__file__).parent / "tests" / "polybench" / "stencils" / "adi"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "FOR": 20,
            "CUDAOffloading": 30,
            "MAP": 10,
            "CUDA": 10,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 3,
            "SEQUENTIAL": 10,
            "CUDAOffloading": 26,
            "MAP": 10,
            "CUDA": 10,
            "FOR": 7,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "REDUCE": 3,
            "SEQUENTIAL": 7,
            "CUDAOffloading": 12,
            "MAP": 6,
            "CUDA": 6,
            "FOR": 4,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 12,
            "MAP": 3,
            "CUDA": 3,
            "FOR": 2,
            "SEQUENTIAL": 5,
            "REDUCE": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 12,
            "MAP": 6,
            "CUDA": 6,
            "FOR": 3,
            "SEQUENTIAL": 6,
            "REDUCE": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
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
        verification={
            "sdfgs": 8,
            "CUDAOffloading": 2,
            "MAP": 2,
            "CUDA": 2,
            "FOR": 5,
            "SEQUENTIAL": 8,
            "REDUCE": 3,
        },
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
        "cuda",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e6,
        ),
        sdfg_verification=verifier,
        docc_flags=["-mllvm", "-docc-einsum"],
    )
    return runner.run(timeout=120)
