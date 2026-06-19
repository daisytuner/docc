import os
import subprocess

from pathlib import Path

import pytest

from test_runner import SDFGVerification


@pytest.mark.parametrize("opt_level", ["-O0", "-O1", "-O2", "-O3"])
def test_static_inline(opt_level):
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "static_inline"
    output_path = benchmark_path / f"static_inline_{opt_level.replace('-', '')}.out"
    cmd = [
        "docc",
        "-g",
        opt_level,
        str(benchmark_path / "main.c"),
        str(benchmark_path / "square.c"),
        str(benchmark_path / "cube.c"),
        "-o",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path), "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    square_res = stdout.splitlines()[0]
    cube_res = stdout.splitlines()[1]
    assert square_res == "Square: 4.000000"
    assert cube_res == "Cube: 8.000000"


@pytest.mark.parametrize("opt_level", ["-O0", "-O1", "-O2", "-O3"])
def test_static_global(opt_level):
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "static_global"
    output_path = benchmark_path / f"static_global_{opt_level.replace('-', '')}.out"
    cmd = [
        "docc-cpp",
        "-g",
        opt_level,
        str(benchmark_path / "main.cpp"),
        str(benchmark_path / "square.cpp"),
        str(benchmark_path / "cube.cpp"),
        "-o",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path), "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    square_res = stdout.splitlines()[0]
    cube_res = stdout.splitlines()[1]
    assert square_res == "Square: 4"
    assert cube_res == "Cube: 8"

def test_memcpy():
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "memory_lto"
    output_path = benchmark_path / f"memcpy.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        str(benchmark_path / "memcpy.c"),
        "-o",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    success = stdout.splitlines()[0]
    assert success == "Success: 1522756.000000 == 1522756.000000"


def test_device_transfers():
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "memory_lto"
    output_path = benchmark_path / f"device_transfers.out"
    cmd = [
        "docc",
        "-mllvm",
        "-docc-tune=cuda",
        "-g",
        "-O3",
        str(benchmark_path / "device_transfers.c"),
        str(benchmark_path / "device_transfers_lib.c"),
        "-o",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    success = stdout.splitlines()[0]
    assert success == "Success: ptr1[100] = 10000.000000"


def test_device_transfers_lib():
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "memory_lto"
    output_path_lib = benchmark_path / f"libdevice_transfers.so"
    cmd_lib = [
        "docc",
        "-mllvm",
        "-docc-tune=cuda",
        "-fPIC",
        "-shared",
        "-g",
        "-O3",
        str(benchmark_path / "device_transfers_lib.c"),
        "-o",
        str(output_path_lib),
    ]
    process = subprocess.Popen(
        cmd_lib,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    output_path = benchmark_path / f"device_transfers.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        str(benchmark_path / "device_transfers.c"),
        "-o",
        str(output_path),
        "-L" + str(benchmark_path),
        "-l" + "device_transfers"
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    os.environ["LD_LIBRARY_PATH"] = str(benchmark_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    success = stdout.splitlines()[0]
    assert success == "Success: ptr1[100] = 10000.000000"


def test_long_name():
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "names"
    output_path = benchmark_path / f"long_name.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        str(benchmark_path / "long_name.c"),
        "-o",
        str(output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    success = stdout.splitlines()[0]
    assert success == "Success: data[10] = 20.000000"

@pytest.mark.xfail(reason="Compilation segfaults")
def test_transfer_minimization_external():
    # Compile lib
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "transfer_minimization"
    output_path_lib = benchmark_path / "libvecadd.so"
    cmd = [
        "clang-19",
        "-fPIC",
        "-shared",
        "-g",
        "-O3",
        str(benchmark_path / "vecadd.c"),
        "-o",
        str(output_path_lib),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    # Compile test
    os.environ["DOCC_OPT_REPORT"] = "1"
    output_path = benchmark_path / "test_external.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        str(benchmark_path / "test_external.c"),
        "-o",
        str(output_path),
        "-L" + str(benchmark_path),
        "-l" + "vecadd"
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    verification = SDFGVerification({
        "sdfgs": 1,
        "ExternalOffloading": 11,
        "Call": 6,
    })
    verification.verify(stderr)

    # Run test
    os.environ["LD_LIBRARY_PATH"] = str(benchmark_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0
    assert "Correct" in stdout.splitlines()

@pytest.mark.xfail(reason="Verifier changed")
def test_transfer_minimization_cuda():
    # Compile test
    os.environ["DOCC_OPT_REPORT"] = "1"
    benchmark_path = Path(__file__).parent / "tests" / "basic" / "transfer_minimization"
    output_path = benchmark_path / "test_cuda.out"
    cmd = [
        "docc",
        "-docc-tune=cuda",
        "-g",
        "-O3",
        str(benchmark_path / "test_cuda.c"),
        "-o",
        str(output_path)
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    verification = SDFGVerification({
        "sdfgs": 1,
        "CUDA": 3,
        "CUDAOffloading": 11,
    })
    verification.verify(stderr)

    # Run test
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0
    assert "Correct" in stdout.splitlines()
