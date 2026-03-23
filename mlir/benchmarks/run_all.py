#!/usr/bin/env python3
"""
Run all torch benchmarks and produce a summary table.

Usage:
    python -m benchmarks.torch.run_all [--n_runs N]

Each benchmark is executed with:
  - torch (torch.compile default backend)
  - docc --target=none        (sequential C)
  - docc --target=sequential   (sequential C, explicit)
  - docc --target=openmp       (OpenMP)
  - docc --target=cuda         (CUDA)
  - docc --target=rocm          (AMD ROCm)

The script prints a table with median runtime per configuration.
"""

import argparse
import importlib
import os
import re
import subprocess
import sys

BENCHMARKS = [
    ("matmul", "benchmarks.torch.layers.matmul", None),
    ("linear", "benchmarks.torch.layers.linear", None),
    ("batchnorm2d", "benchmarks.torch.layers.batchnorm", None),
    ("relu", "benchmarks.torch.layers.relu", None),
    ("conv2d", "benchmarks.torch.layers.conv2d", None),
    ("maxpool2d", "benchmarks.torch.layers.pooling", ["--variant", "maxpool2d"]),
    ("avgpool2d", "benchmarks.torch.layers.pooling", ["--variant", "avgpool2d"]),
]

CONFIGS = [
    ("torch", ["--torch"]),
    ("docc-none", ["--docc", "--target=none"]),
    ("docc-sequential", ["--docc", "--target=sequential"]),
    ("docc-openmp", ["--docc", "--target=openmp"]),
    ("docc-cuda", ["--docc", "--target=cuda"]),
    ("docc-rocm", ["--docc", "--target=rocm"]),
]

TIME_RE = re.compile(r"execution time:\s*([\d.]+)\s*seconds", re.IGNORECASE)


def parse_times(output: str) -> list[float]:
    return [float(m.group(1)) for m in TIME_RE.finditer(output)]


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def run_single(
    module_path: str, extra_args: list[str] | None, config_args: list[str], n_runs: int
) -> float | None:
    cmd = [
        sys.executable,
        "-m",
        module_path,
        *config_args,
        f"--n_runs={n_runs}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",  # docc root
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if result.returncode != 0:
        return None

    times = parse_times(result.stdout)
    if not times:
        return None
    # Drop the first run (warmup / compilation) and take median of the rest
    if len(times) > 1:
        times = times[1:]
    return median(times)


def main():
    parser = argparse.ArgumentParser(description="Run all torch benchmarks")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs per benchmark (default: 5)",
    )
    args = parser.parse_args()

    config_names = [c[0] for c in CONFIGS]
    col_width = max(14, *(len(n) for n in config_names))

    # Header
    header = f"{'benchmark':<20s}"
    for name in config_names:
        header += f"  {name:>{col_width}s}"
    print(header)
    print("-" * len(header))

    # Run benchmarks
    for bench_name, module_path, extra_args in BENCHMARKS:
        row = f"{bench_name:<20s}"
        for config_name, config_args in CONFIGS:
            sys.stdout.flush()
            t = run_single(module_path, extra_args, config_args, args.n_runs)
            if t is not None:
                cell = f"{t:.4f}s"
            else:
                cell = "FAIL"
            row += f"  {cell:>{col_width}s}"
        print(row)
        sys.stdout.flush()

    print()


if __name__ == "__main__":
    main()
