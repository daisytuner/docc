import subprocess
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, SDFGVerification


def evaluate(
    reference_file: Path, test_file: Path, args, max_ulps, dtype=np.float64
) -> float:
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

        row = np.fromstring(line, dtype=dtype, sep=" ")
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

        row = np.fromstring(line, dtype=dtype, sep=" ")
        test_arrays[current_array].append(row)

    if not reference_arrays["output"]:
        return

    reference_arrays["output"] = np.concatenate(reference_arrays["output"])
    test_arrays["output"] = np.concatenate(test_arrays["output"])

    for array in reference_arrays:
        tol = max_ulps * np.abs(np.spacing(reference_arrays[array]))
        assert np.all(np.abs(test_arrays[array] - reference_arrays[array]) <= tol)


@pytest.mark.skip(reason="Timeout & Verifier changed")
def test_bplustree(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "b+tree" / "main.c"
    verifier = SDFGVerification(
        verification={
            "sdfgs": 80,
            "FOR": 39,
            "Malloc": 14,
            "WHILE": 103,
            "Free": 31,
            "MAP": 4,
            "SEQUENTIAL": 4,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "b+tree"
            / "kernel"
            / "kernel_cpu.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "b+tree"
            / "kernel"
            / "kernel_cpu_2.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "b+tree"
            / "util"
            / "timer"
            / "timer.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "b+tree"
            / "util"
            / "num"
            / "num.c",
        ],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                "file",
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "b+tree"
                        / "mil.data"
                    ).absolute()
                ),
                "command",
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "b+tree"
                        / "command.data"
                    ).absolute()
                ),
            ],
            dtype=np.int64,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_backprop(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "backprop" / "backprop.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 25,
            "FOR": 56,
            "Call": 60,
            "Unreachable": 2,
            "WHILE": 2,
            "Free": 6,
            "Malloc": 5,
            "CUDA": 2,
            "CUDAOffloading": 4,
            "MAP": 15,
            "SEQUENTIAL": 13,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "backprop"
            / "backprop_kernel.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "backprop" / "facetrain.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "backprop" / "imagenet.c",
        ],
        partial(evaluate, max_ulps=1e2, args=["1024"], dtype=np.float32),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_bfs(compiler="clang++-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "bfs" / "bfs.cpp"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 3,
            "Malloc": 6,
            "WHILE": 2,
            "FOR": 6,
            "CUDAOffloading": 2,
            "MAP": 2,
            "SEQUENTIAL": 1,
            "CUDA": 1,
            "Free": 6,
            "Unreachable": 1,
            "Call": 20,
        }
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "bfs"
                        / "graph1MW_6.data"
                    ).absolute()
                )
            ],
            dtype=np.int64,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_cfd(compiler="clang++-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "cfd" / "euler3d_cpu.cpp"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 6,
            "ExternalOffloading": 6,
            "CUDAOffloading": 36,
            "MAP": 5,
            "SEQUENTIAL": 1,
            "FOR": 19,
            "CUDA": 4,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e8,
            args=[
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "cfd"
                        / "fvcorr.domn.097K.data"
                    ).absolute()
                )
            ],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
        docc_flags=["-docc-offload-unknown-sizes"],
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed & Output incorrect")
def test_heartwall(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "heartwall" / "main.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 50,
            "Call": 96,
            "Unreachable": 5,
            "CUDAOffloading": 2,
            "MAP": 4,
            "Malloc": 27,
            "FOR": 51,
            "SEQUENTIAL": 3,
            "CUDA": 1,
            "Free": 37,
            "WHILE": 70,
            "Memcpy": 2,
            "Memset": 6,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-I"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "openmp" / "heartwall" / "AVI/"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "heartwall"
            / "AVI"
            / "avilib.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "heartwall"
            / "AVI"
            / "avimod.c",
        ],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "heartwall"
                        / "test.avi.data"
                    ).absolute()
                ),
                "3",
                "1",
            ],
            dtype=np.int64,
        ),
        sdfg_verification=verifier,
        docc_flags=["-docc-offload-unknown-sizes"],
    )
    return runner.run(timeout=350)

@pytest.mark.skip(reason="Test is flaky, needs investigation")
def test_hotspot(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "hotspot" / "hotspot_openmp.cpp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "Free": 2, "FOR": 8, "Calloc": 3, "WHILE": 3},
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                "1024",
                "1024",
                "10",
                "1",
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "hotspot"
                        / "temp_1024.data"
                    ).absolute()
                ),
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "hotspot"
                        / "power_1024.data"
                    ).absolute()
                ),
                "/dev/null",
            ],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


def test_hotspot3D(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "hotspot3D" / "3D.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 8,
            "Malloc": 1,
            "Free": 3,
            "Calloc": 4,
            "WHILE": 2,
            "FOR": 17,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                "512",
                "8",
                "100",
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "hotspot3D"
                        / "power_512x8.data"
                    ).absolute()
                ),
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "hotspot3D"
                        / "temp_512x8.data"
                    ).absolute()
                ),
                "/dev/null",
            ],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed & Output incorrect")
def test_kmeans(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "kmeans" / "kmeans.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 6,
            "Free": 10,
            "MAP": 4,
            "CUDA": 1,
            "CUDAOffloading": 2,
            "Calloc": 2,
            "WHILE": 11,
            "SEQUENTIAL": 3,
            "FOR": 16,
            "Malloc": 6,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "kmeans"
            / "kmeans_clustering.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "kmeans" / "cluster.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "kmeans" / "getopt.c",
        ],
        partial(
            evaluate,
            max_ulps=1e2,
            args=[
                "-i",
                str(
                    (
                        Path(__file__).parent
                        / "tests"
                        / "rodinia"
                        / "data"
                        / "kmeans"
                        / "kdd_cup.data"
                    ).absolute()
                ),
            ],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_lavaMD(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lavaMD" / "main.c"

    verifier = SDFGVerification(
        verification={"sdfgs": 4, "FOR": 21, "WHILE": 5, "Malloc": 8, "Free": 8},
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "lavaMD"
            / "kernel"
            / "kernel_cpu.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "lavaMD"
            / "util"
            / "num"
            / "num.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "lavaMD"
            / "util"
            / "timer"
            / "timer.c",
        ],
        partial(evaluate, max_ulps=1e2, args=["-boxes1d", "10"], dtype=np.float64),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)

@pytest.mark.skip(reason="Crashes on execution")
def test_lud(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lud" / "lud.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 14,
            "WHILE": 3,
            "Free": 9,
            "CUDAOffloading": 10,
            "MAP": 11,
            "CUDA": 4,
            "Malloc": 2,
            "SEQUENTIAL": 7,
            "FOR": 40,
        }
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lud" / "lud_omp.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lud" / "common.c",
        ],
        partial(evaluate, max_ulps=1e2, args=["-s", "4096", "-v"], dtype=np.float32),
        sdfg_verification=verifier,
        docc_flags=["-docc-offload-unknown-sizes"],
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Output incorrect")
def test_nw(compiler="clang++-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "nw" / "nw.cpp"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 25,
            "FOR": 54,
            "Free": 6,
            "MAP": 15,
            "CPU_PARALLEL": 11,
            "SEQUENTIAL": 4,
            "Malloc": 7,
            "WHILE": 4,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(evaluate, max_ulps=1e2, args=["16384", "10", "1"], dtype=np.int64),
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_particlefilter(compiler="clang-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "particlefilter"
        / "particlefilter.c"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 17,
            "Free": 2,
            "Malloc": 4,
            "CUDA": 11,
            "WHILE": 5,
            "CUDAOffloading": 30,
            "MAP": 18,
            "SEQUENTIAL": 7,
            "FOR": 56,
        }
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-std=gnu89",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-w",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=["-x", "2048", "-y", "2048", "-z", "100", "-np", "100"],
            dtype=np.float64,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed & Output incorrect")
def test_pathfinder(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "pathfinder" / "pathfinder.cpp"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 1,
            "FOR": 7,
            "MAP": 2,
            "CUDA": 1,
            "SEQUENTIAL": 1,
            "CUDAOffloading": 4,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(evaluate, max_ulps=1e2, args=["65536", "1000"], dtype=np.int64),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed")
def test_srad(compiler="clang++-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "srad" / "srad.cpp"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 3,
            "Malloc": 11,
            "CUDAOffloading": 12,
            "MAP": 7,
            "FOR": 16,
            "SEQUENTIAL": 4,
            "CUDA": 3,
            "Free": 11,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=["2048", "2048", "0", "127", "0", "127", "1", "0.5", "2"],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Verifier changed & Output incorrect")
def test_streamcluster(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "streamcluster"
        / "streamcluster.cpp"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 25,
            "MAP": 3,
            "SEQUENTIAL": 2,
            "CUDA": 1,
            "CUDAOffloading": 4,
            "WHILE": 28,
            "FOR": 45,
            "Calloc": 3,
            "Malloc": 7,
            "Free": 8,
        },
    )
    runner = TestRunner(
        "Rodinia",
        test_case,
        "docc-cpp",
        compiler,
        [
            "-O3",
            "-g",
            "-fopenmp",
            "-DRODINIA_DUMP_ARRAYS",
            "-D_MY_IS_NOT_CUDA_",
            "-D_MY_IS_OPENMP",
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent / "tests" / "rodinia" / "common" / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "cuda",
        [],
        partial(
            evaluate,
            max_ulps=1e2,
            args=["2", "3", "256", "65536", "65536", "1000", "none", "/dev/null", "1"],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)
