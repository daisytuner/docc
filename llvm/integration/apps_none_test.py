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

    ref_final_residual = float(header.split(":")[1].strip())

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

    test_final_residual = float(header.split(":")[1].strip())

    assert test_final_residual == ref_final_residual


def test_HPCCG():
    test_case = Path(__file__).parent / "tests" / "apps" / "HPCCG" / "main.cpp"
    verifier = SDFGVerification(
        verification={"sdfgs": 19, "FOR": 14, "SEQUENTIAL": 20, "MAP": 6, "WHILE": 6},
    )
    runner = TestRunner(
        "HPCCG",
        test_case,
        "docc-cpp",
        "clang++-19",
        ["-O3", "-g", "-lm"],
        "none",
        [
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "compute_residual.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "ddot.cpp",
            Path(__file__).parent / "tests" / "apps" / "HPCCG" / "generate_matrix.cpp",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "HPCCG"
            / "HPC_Sparse_Matrix.cpp",
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
    )
    runner.run()


def evaluate_lulesh(reference_file: Path, test_file: Path, args) -> float:
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
    while not header.startswith("Final Origin Energy"):
        header = data.pop(0).strip()

    ref_origin_energy = float(header.split("=")[1].strip())

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
    found = False
    while not header.startswith("Final Origin Energy"):
        header = data.pop(0).strip()
        found = True

    assert found, "Final Origin Energy not found in output"

    test_origin_energy = float(header.split("=")[1].strip())

    print(test_origin_energy, ref_origin_energy)
    assert test_origin_energy == ref_origin_energy


def test_LULESH():
    test_case = Path(__file__).parent / "tests" / "apps" / "LULESH" / "lulesh.cc"

    verifier = SDFGVerification(
        verification={"sdfgs": 29, "FOR": 22, "SEQUENTIAL": 36, "WHILE": 24, "MAP": 14}
    )
    runner = TestRunner(
        "Apps",
        test_case,
        "docc-cpp",
        "clang++-19",
        [
            "-O3",
            "-fopenmp",
            "-g",
            "-DUSE_MPI=0",
            "-DUSE_OMP=0",
            "-lm",
        ],
        "none",
        [
            Path(__file__).parent / "tests" / "apps" / "LULESH" / "lulesh-comm.cc",
            Path(__file__).parent / "tests" / "apps" / "LULESH" / "lulesh-init.cc",
            Path(__file__).parent / "tests" / "apps" / "LULESH" / "lulesh-util.cc",
            Path(__file__).parent / "tests" / "apps" / "LULESH" / "lulesh-viz.cc",
        ],
        partial(
            evaluate_lulesh,
            args=["-s", "30", "-i", "10"],
        ),
        sdfg_verification=verifier,
    )
    runner.run()


def evaluate_miniFE(reference_file: Path, test_file: Path, args) -> float:
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
    while not header.startswith("Final Resid Norm"):
        header = data.pop(0).strip()

    ref_final_residual = float(header.split(":")[1].strip())

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
    while not header.startswith("Final Resid Norm"):
        header = data.pop(0).strip()

    test_final_residual = float(header.split(":")[1].strip())

    assert abs(test_final_residual - ref_final_residual) < 10e-8


@pytest.mark.parametrize(
    "data_layout, precision",
    [
        pytest.param(
            "MINIFE_CSR_MATRIX",
            "float",
            marks=pytest.mark.xfail(reason="Compilation segfaults"),
        ),
        pytest.param(
            "MINIFE_ELL_MATRIX",
            "float",
            marks=pytest.mark.xfail(reason="Compilation segfaults"),
        ),
    ],
)
def test_miniFE(data_layout, precision):
    test_case = Path(__file__).parent / "tests" / "apps" / "miniFE" / "src" / "main.cpp"
    verifier_csr = SDFGVerification(
        verification={
            "sdfgs": 76,
            "MAP": 25,
            "FOR": 183,
            "WHILE": 57,
            "SEQUENTIAL": 25,
        }
    )
    verifier_ell = SDFGVerification(
        verification={
            "sdfgs": 73,
            "MAP": 15,
            "FOR": 128,
            "WHILE": 45,
            "SEQUENTIAL": 15,
        }
    )
    if data_layout == "MINIFE_CSR_MATRIX":
        verifier = verifier_csr
    else:
        verifier = verifier_ell

    runner = TestRunner(
        "Apps",
        test_case,
        "docc-cpp",
        "clang++-19",
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "apps" / "miniFE" / "src").absolute()
            ),
            "-I"
            + str(
                (
                    Path(__file__).parent / "tests" / "apps" / "miniFE" / "utils"
                ).absolute()
            ),
            "-I"
            + str(
                (Path(__file__).parent / "tests" / "apps" / "miniFE" / "fem").absolute()
            ),
            "-DMINIFE_SCALAR=" + precision,
            "-DMINIFE_LOCAL_ORDINAL=int",
            "-DMINIFE_GLOBAL_ORDINAL=int",
            "-D" + data_layout,
            "-UHAVE_MPI",
            "-lm",
        ],
        "none",
        [
            Path(__file__).parent
            / "tests"
            / "apps"
            / "miniFE"
            / "utils"
            / "param_utils.cpp",
            Path(__file__).parent / "tests" / "apps" / "miniFE" / "utils" / "utils.cpp",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "miniFE"
            / "utils"
            / "mytimer.cpp",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "miniFE"
            / "src"
            / "YAML_Element.cpp",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "miniFE"
            / "src"
            / "YAML_Doc.cpp",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "miniFE"
            / "basic"
            / "BoxPartition.cpp",
        ],
        partial(
            evaluate_miniFE,
            args=[],
        ),
        sdfg_verification=verifier,
        docc_flags=["-docc-lower-invoke"],
    )
    runner.run(timeout=1500)


def evaluate_miniAMR2(reference_file: Path, test_file: Path, args) -> float:
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
    while header != "BEGIN DUMP OUTPUT":
        header = data.pop(0)
    assert header == "BEGIN DUMP OUTPUT"

    reference_arrays = {"output": []}
    current_array = "output"
    for line in data:
        if line == "END DUMP OUTPUT":
            break

        row = np.fromstring(line, dtype=np.float64, sep=" ")
        reference_arrays[current_array].append(row)

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
    while header != "BEGIN DUMP OUTPUT":
        header = data.pop(0)
    assert header == "BEGIN DUMP OUTPUT"

    test_arrays = {"output": []}
    current_array = "output"
    for line in data:
        if line == "END DUMP OUTPUT":
            break

        row = np.fromstring(line, dtype=np.float64, sep=" ")
        test_arrays[current_array].append(row)

    if not reference_arrays["output"]:
        return

    reference_arrays["output"] = np.concatenate(reference_arrays["output"])
    test_arrays["output"] = np.concatenate(test_arrays["output"])

    for array in reference_arrays:
        assert np.array_equal(
            reference_arrays[array], test_arrays[array], equal_nan=True
        )


def test_miniAMR2():
    test_case = Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "main.c"

    verifier = SDFGVerification(
        verification={"sdfgs": 48, "WHILE": 59, "FOR": 21, "SEQUENTIAL": 22, "MAP": 1}
    )
    runner = TestRunner(
        "Apps",
        test_case,
        "docc",
        "clang-19",
        ["-O3", "-g", "-fopenmp", "-latomic", "-DMANTEVO_DUMP_ARRAYS"],
        "none",
        [
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "block.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "calc.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "params.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "plot.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "profile.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "queue.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "refine.c",
            Path(__file__).parent / "tests" / "apps" / "miniAMR2" / "stencil.c",
        ],
        partial(
            evaluate_miniAMR2,
            args=[
                "--nx",
                "16",
                "--ny",
                "16",
                "--nz",
                "16",
            ],
        ),
        sdfg_verification=verifier,
    )
    runner.run()


def evaluate_cloudsc(reference_file: Path, test_file: Path, args) -> float:
    cmd = [reference_file] + args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    cmd = [test_file] + args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0


@pytest.mark.skip(reason="Compile time")
def test_cloudsc():
    test_case = (
        Path(__file__).parent / "tests" / "apps" / "cloudsc_c" / "dwarf_cloudsc.c"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 48,
            "MAP": 3,
            "FOR": 57,
            "WHILE": 24,
            "SEQUENTIAL": 3,
            "Malloc": 13,
            "Free": 7,
        }
    )
    runner = TestRunner(
        "Apps",
        test_case,
        "docc",
        "clang-19",
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DHAVE_HDF5",
            "-I/usr/include/hdf5/serial",
            "-lhdf5_serial",
        ],
        "none",
        [
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "cloudsc_c.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "cloudsc_driver.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "cloudsc_validate.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "load_state.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "mycpu.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "yoecldp_c.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "yoethf_c.c",
            Path(__file__).parent
            / "tests"
            / "apps"
            / "cloudsc_c"
            / "cloudsc"
            / "yomcst_c.c",
        ],
        partial(
            evaluate_cloudsc,
            args=[
                "1",
                "8192",
                "128",
            ],
        ),
        sdfg_verification=verifier,
    )
    runner.run()
