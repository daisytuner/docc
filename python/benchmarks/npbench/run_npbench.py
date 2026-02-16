#!/usr/bin/env python3
"""
Script to run all polybench benchmarks with different targets and collect stats.
"""

import argparse
import csv
import importlib
import io
import re
import sys
import signal
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import docc

# Add the python directory to the path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from benchmarks.npbench.harness import run_benchmark


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Benchmark execution timed out")


def run_benchmark_with_target(
    module_name,
    benchmark_name,
    initialize_func,
    kernel_func,
    parameters,
    target,
    size="S",
    n_runs=10,
    use_numpy=False,
    remote_tuning=False,
):
    """Run a single benchmark with a specific target and collect stats."""

    # Validate size parameter
    if size not in parameters:
        print(
            f"Warning: Size '{size}' not found in parameters. Available: {list(parameters.keys())}"
        )
        size = "S" if "S" in parameters else list(parameters.keys())[0]
        print(f"Using size: {size}")

    try:
        # Create args object to pass to run_benchmark
        if use_numpy:
            args = SimpleNamespace(
                size=size, docc=False, numpy=True, target="none", n_runs=n_runs, remote_tuning=False
            )
        else:
            args = SimpleNamespace(
                size=size, docc=True, numpy=False, target=target, n_runs=n_runs, remote_tuning=remote_tuning
            )

        # Capture the output from run_benchmark (silence stdout and stderr)
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        # Set up timeout (200 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(200 * n_runs)  # Total timeout based on number of runs

        try:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                run_benchmark(
                    initialize_func, kernel_func, parameters, benchmark_name, args=args
                )
        finally:
            # Cancel the alarm
            signal.alarm(0)

        output = output_buffer.getvalue()

        # Parse timing information from output
        if use_numpy:
            # For numpy, look for "Numpy execution time: X seconds"
            numpy_time_match = re.search(
                r"Numpy execution time: ([\d.]+) seconds", output
            )
            first_execution_time = (
                float(numpy_time_match.group(1)) if numpy_time_match else None
            )
            cached_times = re.findall(r"Numpy execution time: ([\d.]+) seconds", output)
            cached_times_float = [float(t) for t in cached_times]
            stats = {}
        else:
            # For docc, look for "Docc execution time" and stats
            first_time_match = re.search(
                r"Docc execution time: ([\d.]+) seconds", output
            )
            cached_times = re.findall(
                r"Docc execution time \(cached\): ([\d.]+) seconds", output
            )

            first_execution_time = (
                float(first_time_match.group(1)) if first_time_match else None
            )
            cached_times_float = [float(t) for t in cached_times]

            # Parse stats from output
            stats_match = re.search(r"\{(.+?)\}", output)
            stats = {}
            if stats_match:
                stats_str = "{" + stats_match.group(1) + "}"
                try:
                    stats = eval(stats_str)
                except:
                    pass

        # Add metadata with all timing information
        result = {
            "module": module_name,
            "benchmark": benchmark_name,
            "target": target if not use_numpy else "numpy",
            "size": size,
            "first_execution_time": first_execution_time,
            "avg_cached_time": (
                np.mean(cached_times_float) if cached_times_float else None
            ),
            "min_cached_time": (
                np.min(cached_times_float) if cached_times_float else None
            ),
            "max_cached_time": (
                np.max(cached_times_float) if cached_times_float else None
            ),
            "success": True,
            "error": None,
        }

        # Add individual cached run times
        for i, cached_time in enumerate(cached_times_float):
            result[f"cached_time_{i}"] = cached_time

        # Add all stats
        result.update(stats)

        return result

    except TimeoutError as e:
        target_str = "numpy" if use_numpy else target
        error_msg = f"Benchmark timed out after 200 seconds"
        print(f"Timeout running {benchmark_name} with target {target_str}: {error_msg}")
        return {
            "module": module_name,
            "benchmark": benchmark_name,
            "target": target_str,
            "size": size,
            "first_execution_time": None,
            "avg_cached_time": None,
            "min_cached_time": None,
            "max_cached_time": None,
            "success": False,
            "error": error_msg,
        }
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        target_str = "numpy" if use_numpy else target
        print(f"Error running {benchmark_name} with target {target_str}: {error_msg}")
        return {
            "module": module_name,
            "benchmark": benchmark_name,
            "target": target_str,
            "size": size,
            "first_execution_time": None,
            "avg_cached_time": None,
            "min_cached_time": None,
            "max_cached_time": None,
            "success": False,
            "error": error_msg,
        }


def get_all_polybench_tests():
    """Discover all test modules in polybench and other benchmark directories."""
    benchmark_dirs = ["polybench", "cavity_flow", "go_fast", "spmv", "weather_stencils"]

    benchmarks = []

    for dir_name in benchmark_dirs:
        bench_dir = script_dir / "benchmarks" / "npbench" / dir_name

        if not bench_dir.exists():
            print(f"Warning: Directory {bench_dir} does not exist, skipping...")
            continue

        print(f"\nScanning directory: {dir_name}")
        test_files = sorted(bench_dir.glob("test_*.py"))

        for test_file in test_files:

            module_name = f"benchmarks.npbench.{dir_name}.{test_file.stem}"
            benchmark_name = test_file.stem.replace("test_", "")

            # Check if the test function is marked with @pytest.mark.skip()
            # and extract available targets from pytest.mark.parametrize
            with open(test_file, "r") as f:
                content = f.read()
                if "@pytest.mark.skip(" in content:
                    print(f"Skipping {benchmark_name}: marked with @pytest.mark.skip()")
                    continue

                # Extract available targets from @pytest.mark.parametrize("target", [...])
                available_targets = None
                parametrize_match = re.search(
                    r'@pytest\.mark\.parametrize\(\s*["\']target["\']\s*,\s*\[(.*?)\]',
                    content,
                    re.DOTALL,
                )
                if parametrize_match:
                    targets_str = parametrize_match.group(1)
                    # Extract all non-commented target strings
                    target_matches = re.findall(r'["\'](\w+)["\']', targets_str)
                    available_targets = target_matches
                    # Check for commented out targets
                    commented_targets = re.findall(r'#\s*["\'](\w+)["\']', targets_str)
                    if commented_targets:
                        # Remove commented targets from available list
                        available_targets = [
                            t for t in available_targets if t not in commented_targets
                        ]
                    print(
                        f"  {benchmark_name}: available targets = {available_targets}"
                    )

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Check if it has the required functions
                if (
                    hasattr(module, "initialize")
                    and hasattr(module, "kernel")
                    and hasattr(module, "PARAMETERS")
                ):
                    benchmarks.append(
                        {
                            "module_name": module_name,
                            "benchmark_name": benchmark_name,
                            "initialize": module.initialize,
                            "kernel": module.kernel,
                            "parameters": module.PARAMETERS,
                            "available_targets": available_targets,
                        }
                    )
                else:
                    print(f"Skipping {module_name}: missing required attributes")

            except Exception as e:
                print(f"Failed to import {module_name}: {e}")

    return benchmarks


def main():
    """Main function to run all benchmarks and save results."""

    parser = argparse.ArgumentParser(description="Run polybench benchmark suite")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of cached runs per benchmark (default: 10)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["none", "sequential", "openmp", "cuda", "numpy"],
        help="Targets to run (default: none sequential openmp cuda numpy). Use 'numpy' for numpy baseline.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="S",
        help="Size parameter for benchmarks (default: S)",
    )
    parser.add_argument(
        "--remote-tuning",
        action="store_true",
        default=False,
        help="Enable remote tuning via RPC (default: False)",
    )
    args = parser.parse_args()

    n_runs = args.n_runs
    size = args.size
    targets = args.targets
    remote_tuning = args.remote_tuning

    print(f"Configuration: n_runs={n_runs}, targets={targets}, size={size}, remote_tuning={remote_tuning}")

    # Discover all benchmarks
    print("Discovering benchmarks...")
    benchmarks = get_all_polybench_tests()
    print(f"Found {len(benchmarks)} benchmarks")

    # Collect all results
    all_results = []

    # Calculate total runs by checking available targets for each benchmark
    total_runs = 0
    for benchmark in benchmarks:
        available_targets = benchmark.get("available_targets")
        if available_targets is not None:
            benchmark_targets = [
                t for t in targets if t == "numpy" or t in available_targets
            ]
        else:
            benchmark_targets = targets
        total_runs += len(benchmark_targets)

    current_run = 0

    for benchmark in benchmarks:
        print(f"\n{'='*80}")
        print(f"Running benchmark: {benchmark['benchmark_name']}")

        # Filter targets based on what's available for this benchmark
        available_targets = benchmark.get("available_targets")
        if available_targets is not None:
            # Filter the requested targets to only those available in the pytest
            # Always keep numpy as an option even if not in pytest targets
            benchmark_targets = [
                t for t in targets if t == "numpy" or t in available_targets
            ]
            if len(benchmark_targets) < len(targets):
                unavailable = [t for t in targets if t not in benchmark_targets]
                print(f"Note: Targets {unavailable} not available for this benchmark")
        else:
            # If we couldn't parse targets, use all requested targets
            benchmark_targets = targets

        print(f"Available targets: {benchmark_targets}")
        print(f"{'='*80}")

        for target in benchmark_targets:
            current_run += 1
            # Check if this is a numpy baseline run
            is_numpy = target == "numpy"
            print(f"\n[{current_run}/{total_runs}] Target: {target}")

            result = run_benchmark_with_target(
                benchmark["module_name"],
                benchmark["benchmark_name"],
                benchmark["initialize"],
                benchmark["kernel"],
                benchmark["parameters"],
                target,
                size=size,
                n_runs=n_runs,
                use_numpy=is_numpy,
                remote_tuning=remote_tuning,
            )

            all_results.append(result)

            # Print summary
            if result["success"]:
                print(
                    f"  ✓ Success - First: {result['first_execution_time']:.6f}s, Avg Cached: {result['avg_cached_time']:.6f}s"
                )
                stats_summary = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in [
                        "module",
                        "benchmark",
                        "target",
                        "size",
                        "first_execution_time",
                        "avg_cached_time",
                        "min_cached_time",
                        "max_cached_time",
                        "success",
                        "error",
                    ]
                    and not k.startswith("cached_time_")
                }
                print(f"  Stats: {stats_summary}")
            else:
                print(f"  ✗ Failed - Error: {result['error']}")

    # Save results to CSV
    output_file = script_dir / "polybench_results.csv"

    if all_results:
        # Get all unique keys from all results
        all_keys = set()
        for result in all_results:
            all_keys.update(result.keys())

        fieldnames = [
            "module",
            "benchmark",
            "target",
            "size",
            "first_execution_time",
            "avg_cached_time",
            "min_cached_time",
            "max_cached_time",
            "success",
            "error",
        ]
        # Add cached time fields in order
        cached_time_fields = sorted(
            [k for k in all_keys if k.startswith("cached_time_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        fieldnames.extend(cached_time_fields)
        # Add stat fields
        stat_fields = sorted(
            [
                k
                for k in all_keys
                if k not in fieldnames and not k.startswith("cached_time_")
            ]
        )
        fieldnames.extend(stat_fields)

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"Total benchmarks: {len(benchmarks)}")
        print(f"Total runs: {len(all_results)}")
        successful = sum(1 for r in all_results if r["success"])
        failed = len(all_results) - successful
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*80}")
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()
