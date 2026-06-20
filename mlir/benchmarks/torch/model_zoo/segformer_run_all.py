#!/usr/bin/env python3
"""
Run every SegFormer benchmark configuration and print an overview.

Each variant defined in ``segformer.BENCHMARKS`` is executed once via the
standard benchmark harness, and the runtime of the final run is recorded.

Usage (arguments mirror the original ``segformer.py`` invocation):

    python -m benchmarks.torch.model_zoo.segformer_run_all \
        --batch-size 1 --device docc --target cuda --n_runs 10

The script prints a table with the final runtime per variant.
"""

import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# benchmarks/torch/model_zoo -> mlir (root used to resolve the module path)
MLIR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SEGFORMER_MODULE = "benchmarks.torch.model_zoo.segformer"

TIME_RE = re.compile(r"execution time:\s*([\d.]+)\s*seconds", re.IGNORECASE)


def _load_variants() -> list[str]:
    if MLIR_DIR not in sys.path:
        sys.path.insert(0, MLIR_DIR)
    from benchmarks.torch.model_zoo.segformer import BENCHMARKS

    return [v for v in BENCHMARKS.keys() if v != "default"]


def _parse_times(output: str) -> list[float]:
    return [float(m.group(1)) for m in TIME_RE.finditer(output)]


def run_variant(
    variant: str,
    batch_size: int,
    device: str,
    target: str,
    n_runs: int,
    timeout: float,
) -> float | None:
    """Run a single variant and return the average runtime in seconds.

    The first run (warmup / compilation) is excluded from the average.
    """
    cmd = [
        sys.executable,
        "-m",
        SEGFORMER_MODULE,
        "--variant",
        variant,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--target",
        target,
        "--n_runs",
        str(n_runs),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=MLIR_DIR,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if result.returncode != 0:
        return None

    times = _parse_times(result.stdout)
    if not times:
        return None
    # Skip the first run (warmup / compilation) from the average.
    if len(times) > 1:
        times = times[1:]
    return sum(times) / len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all SegFormer benchmarks")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "rocm", "docc"],
        default="cpu",
        help="Device backend to benchmark on",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="none",
        help="Docc compilation target (only used when --device docc)",
    )
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Per-variant timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Run only the given variant(s); may be repeated. Default: all.",
    )
    args = parser.parse_args()

    all_variants = _load_variants()
    if args.variants:
        unknown = [v for v in args.variants if v not in all_variants]
        if unknown:
            parser.error(f"unknown variant(s): {', '.join(unknown)}")
        variants = args.variants
    else:
        variants = all_variants

    backend_label = args.device
    if args.device == "docc":
        backend_label = f"docc_{args.target}"

    print(
        f"SegFormer benchmark overview  |  batch_size={args.batch_size}  "
        f"backend={backend_label}  n_runs={args.n_runs}",
        flush=True,
    )

    name_width = max(len("variant"), *(len(v) for v in variants))
    header = f"{'variant':<{name_width}s}  {'avg runtime':>16s}"
    print(header)
    print("-" * len(header))

    results: dict[str, float | None] = {}
    for variant in variants:
        sys.stdout.flush()
        t = run_variant(
            variant,
            args.batch_size,
            args.device,
            args.target,
            args.n_runs,
            args.timeout,
        )
        results[variant] = t
        cell = f"{t:.6f}s" if t is not None else "FAIL"
        print(f"{variant:<{name_width}s}  {cell:>16s}", flush=True)

    succeeded = sum(1 for t in results.values() if t is not None)
    print("-" * len(header))
    print(f"{succeeded}/{len(variants)} variants completed", flush=True)


if __name__ == "__main__":
    main()
