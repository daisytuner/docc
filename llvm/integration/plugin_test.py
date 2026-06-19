import os
import subprocess

from pathlib import Path


def test_simple_plugin():
    benchmark_path = Path(__file__).parent / "tests" / "plugins" / "simple" / "test"
    output_path = benchmark_path / "simple.out"

    # Build without plugin library
    cmd = [
        "docc-cpp",
        "-g",
        "-O0",
        str(benchmark_path / "main.cpp"),
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

    # Run benchmark without plugin library
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 1
    res = stdout.splitlines()
    assert "foo() was NOT replaced by plugin!" in res

    # Build with plugin library
    cmd = [
        "docc-cpp",
        "-g",
        "-O0",
        "-mllvm",
        "-docc-plugins=libSimplePlugin.so",
        str(benchmark_path / "main.cpp"),
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
    res = stderr.splitlines()
    assert "[DEBUG] [docc] Loaded plugin: simple (0.0.1)" in res

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0
    res = stdout.splitlines()
    assert "foo() was replaced by plugin!" in res
