import subprocess
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from functools import partial

from test_runner import TestRunner, SDFGVerification
import signal


def evaluate(reference_file: Path, test_file: Path, args, dtype=np.float64) -> float:
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
        assert np.array_equal(
            reference_arrays[array], test_arrays[array], equal_nan=True
        )


@pytest.mark.skip(reason="Timeout")
def test_bplustree(compiler="clang-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "b+tree" / "main.c"
    )
    verifier = SDFGVerification(
        verification={
            "sdfgs": 80,
            "FOR": 28,
            "Malloc": 14,
            "WHILE": 113,
            "Free": 31,
            "MAP": 2,
            "SEQUENTIAL": 2,
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
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


def test_backprop(compiler="clang-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "backprop"
        / "backprop.c"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 25, "FOR": 37, "SEQUENTIAL": 52, "WHILE": 6, "MAP": 15}
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "backprop"
            / "backprop_kernel.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "backprop"
            / "facetrain.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "backprop"
            / "imagenet.c",
        ],
        partial(evaluate, args=["1024"], dtype=np.float32),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


def test_bfs(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "bfs" / "bfs.cpp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 3, "WHILE": 7, "MAP": 1, "SEQUENTIAL": 1},
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
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


def test_cfd(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "cfd"
        / "euler3d_cpu.cpp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 6, "MAP": 6, "SEQUENTIAL": 19, "FOR": 13}
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
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
    )

    return runner.run(timeout=240)


@pytest.mark.xfail(reason="Timeout")
def test_heartwall(compiler="clang-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "heartwall" / "main.c"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 50,
            "Malloc": 27,
            "MAP": 7,
            "FOR": 48,
            "SEQUENTIAL": 7,
            "Free": 37,
            "WHILE": 73,
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-I"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "openmp"
                    / "heartwall"
                    / "AVI/"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
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
    )

    return runner.run(timeout=350)


@pytest.mark.skip(reason="Test is flaky, needs investigation")
def test_hotspot(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "hotspot"
        / "hotspot_openmp.cpp"
    )
    verifier = SDFGVerification(
        verification={"sdfgs": 8, "Free": 2, "FOR": 8, "WHILE": 3, "Calloc": 3},
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
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
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "hotspot3D" / "3D.c"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 8, "WHILE": 2, "MAP": 4, "FOR": 13, "SEQUENTIAL": 17}
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
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


@pytest.mark.skip(reason="Reference executable segfaults (SIGSEGV) in CI environment")
def test_kmeans(compiler="clang-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "kmeans" / "kmeans.c"
    )

    verifier = SDFGVerification(
        verification={
            "sdfgs": 6,
            "Free": 10,
            "FOR": 16,
            "MAP": 4,
            "Calloc": 2,
            "Malloc": 6,
            "WHILE": 11,
            "SEQUENTIAL": 4,
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "kmeans"
            / "kmeans_clustering.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "kmeans"
            / "cluster.c",
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "kmeans"
            / "getopt.c",
        ],
        partial(
            evaluate,
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


def test_lavaMD(compiler="clang-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lavaMD" / "main.c"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 4, "FOR": 20, "WHILE": 6, "Malloc": 8, "Free": 8}
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
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
        partial(evaluate, args=["-boxes1d", "10"], dtype=np.float64),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


@pytest.mark.skip(reason="Timeout")
def test_lud(compiler="clang-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lud" / "lud.c"

    verifier = SDFGVerification(
        verification={
            "sdfgs": 14,
            "Free": 9,
            "Malloc": 2,
            "FOR": 40,
            "MAP": 11,
            "WHILE": 3,
            "SEQUENTIAL": 11,
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [
            Path(__file__).parent
            / "tests"
            / "rodinia"
            / "openmp"
            / "lud"
            / "lud_omp.c",
            Path(__file__).parent / "tests" / "rodinia" / "openmp" / "lud" / "common.c",
        ],
        partial(evaluate, args=["-s", "4096", "-v"], dtype=np.float32),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


def test_nw(compiler="clang++-19"):
    test_case = Path(__file__).parent / "tests" / "rodinia" / "openmp" / "nw" / "nw.cpp"

    verifier = SDFGVerification(
        verification={"sdfgs": 7, "WHILE": 5, "FOR": 15, "SEQUENTIAL": 28, "MAP": 13},
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(evaluate, args=["16384", "10", "1"], dtype=np.int64),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


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
            "REDUCE": 1,
            "WHILE": 12,
            "MAP": 19,
            "FOR": 29,
            "SEQUENTIAL": 49,
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
            args=["-x", "2048", "-y", "2048", "-z", "100", "-np", "100"],
            dtype=np.float64,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


def test_pathfinder(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent
        / "tests"
        / "rodinia"
        / "openmp"
        / "pathfinder"
        / "pathfinder.cpp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 1, "FOR": 5, "SEQUENTIAL": 7, "MAP": 2},
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(evaluate, args=["65536", "1000"], dtype=np.int64),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


def test_srad(compiler="clang++-19"):
    test_case = (
        Path(__file__).parent / "tests" / "rodinia" / "openmp" / "srad" / "srad.cpp"
    )

    verifier = SDFGVerification(
        verification={"sdfgs": 3, "MAP": 7, "SEQUENTIAL": 16, "FOR": 9}
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
            args=["2048", "2048", "0", "127", "0", "127", "1", "0.5", "2"],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)


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
        verification={"sdfgs": 25, "MAP": 2, "FOR": 39, "SEQUENTIAL": 41, "WHILE": 32},
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
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_timing.h"
                ).absolute()
            ),
            "-include"
            + str(
                (
                    Path(__file__).parent
                    / "tests"
                    / "rodinia"
                    / "common"
                    / "my_verification.h"
                ).absolute()
            ),
            "-lm",
        ],
        "none",
        [],
        partial(
            evaluate,
            args=["2", "3", "256", "65536", "65536", "1000", "none", "/dev/null", "1"],
            dtype=np.float32,
        ),
        sdfg_verification=verifier,
    )
    return runner.run(timeout=240)
