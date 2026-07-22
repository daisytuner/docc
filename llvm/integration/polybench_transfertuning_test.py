import subprocess
import numpy as np
import pandas as pd

import re
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, TransformationVerification


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

    transformation_verification = TransformationVerification(
        {
            # "RPCNodeTransform": {
            #     "loop_nests": {},
            #     "tuned_loops": 9,
            # }
        }
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 6}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 4}}
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
        ),
        transformation_verification=transformation_verification,
        docc_flags=["-docc-transfer-tune", "-docc-tune=sequential", "-docc-save-temps"],
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

    transformation_verification = TransformationVerification(
        {
            # "RPCNodeTransform": {
            #     "loop_nests": {},
            #     "tuned_loops": 2,
            # }
        }
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
            max_ulps=1e2,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 3}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 3}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {
            # "RPCNodeTransform": {
            #     "loop_nests": {},
            #     "tuned_loops": 4,
            # }
        }
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 4}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 6}}
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
        transformation_verification=transformation_verification,
        docc_flags=["-docc-transfer-tune", "-docc-tune=sequential", "-docc-save-temps"],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 7}}
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
        ),
        transformation_verification=transformation_verification,
        docc_flags=["-docc-transfer-tune", "-docc-tune=sequential", "-docc-save-temps"],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 4}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 4}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 1}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 6}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 5}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 2}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {}, "tuned_loops": 3}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
    )
    return runner.run(timeout=120)


def test_floyd_warshall(compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "floyd-warshall"
    )

    transformation_verification = TransformationVerification({"RPCNodeTransform": {1}})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
    )
    return runner.run(timeout=120)


def test_nussinov(compiler="clang-19", size="MEDIUM_DATASET"):
    benchmark_path = (
        Path(__file__).parent / "tests" / "polybench" / "medley" / "nussinov"
    )

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(verify, dtype=np.int64),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification({})

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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
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

    transformation_verification = TransformationVerification(
        {"RPCNodeTransform": {"loop_nests": {1}, "tuned_loops": 2}}
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
            "-lblas",
        ],
        "sequential",
        [Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"],
        partial(
            verify,
            dtype=np.float64 if datatype == "-DDATA_TYPE_IS_DOUBLE" else np.float32,
        ),
        transformation_verification=transformation_verification,
        docc_flags=[
            "-mllvm",
            "-docc-einsum",
            "-docc-transfer-tune",
            "-docc-tune=sequential",
            "-docc-save-temps",
        ],
    )
    return runner.run(timeout=120)
