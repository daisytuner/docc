#!/usr/bin/env python3
"""
Run every SegFormer benchmark configuration and print an overview.

Each variant defined in ``segformer.BENCHMARKS`` is executed via the standard
benchmark harness for every requested batch size, and the average runtime
(excluding the warmup run) is recorded.

Usage (arguments mirror the original ``segformer.py`` invocation):

    python -m benchmarks.torch.model_zoo.segformer_run_all \
        --batch-size 1 4 16 --device docc --target cuda --n_runs 10

The script prints a table with the average runtime per variant, one column
per batch size.
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
# The stage index is the first number following a stage-prefixed segment, e.g.
# "...block.1.0" -> stage 1, while "...block.0.1" -> stage 0 (the trailing .1 is
# the layer within stage 0, not a stage selector).
STAGE_RE = re.compile(r"(?:path_embeddings|block|linear_c)\.(\d+)")


def _load_variants() -> list[str]:
    if MLIR_DIR not in sys.path:
        sys.path.insert(0, MLIR_DIR)
    from benchmarks.torch.model_zoo.segformer import BENCHMARKS

    return [v for v in BENCHMARKS.keys() if v not in ("default")]


def _variant_stage(variant: str) -> int | None:
    """Return the encoder stage index a variant belongs to, or None."""
    m = STAGE_RE.search(variant)
    return int(m.group(1)) if m else None


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
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1])
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
    parser.add_argument(
        "--stage",
        type=int,
        action="append",
        dest="stages",
        help="Run only the layers within the given encoder stage; may be "
        "repeated. Cannot be combined with --variant.",
    )
    args = parser.parse_args()

    if args.variants and args.stages is not None:
        parser.error("--stage cannot be combined with --variant")

    all_variants = _load_variants()
    if args.variants:
        unknown = [v for v in args.variants if v not in all_variants]
        if unknown:
            parser.error(f"unknown variant(s): {', '.join(unknown)}")
        variants = args.variants
    elif args.stages is not None:
        stages = set(args.stages)
        variants = [v for v in all_variants if _variant_stage(v) in stages]
        if not variants:
            wanted = ", ".join(str(s) for s in sorted(stages))
            parser.error(f"no variants found for stage(s): {wanted}")
    else:
        variants = all_variants

    backend_label = args.device
    if args.device == "docc":
        backend_label = f"docc_{args.target}"

    batch_sizes = args.batch_size
    print(
        f"SegFormer benchmark overview  |  batch_sizes={batch_sizes}  "
        f"backend={backend_label}  n_runs={args.n_runs}",
        flush=True,
    )

    name_width = max(len("variant"), *(len(v) for v in variants))
    col_labels = [f"bs={bs}" for bs in batch_sizes]
    col_width = max(16, *(len(c) for c in col_labels))
    header = f"{'variant':<{name_width}s}"
    for label in col_labels:
        header += f"  {label:>{col_width}s}"
    print(header)
    print("-" * len(header))

    results: dict[str, dict[int, float | None]] = {}
    succeeded = 0
    total = 0
    for variant in variants:
        row = f"{variant:<{name_width}s}"
        results[variant] = {}
        for bs in batch_sizes:
            sys.stdout.flush()
            t = run_variant(
                variant,
                bs,
                args.device,
                args.target,
                args.n_runs,
                args.timeout,
            )
            results[variant][bs] = t
            total += 1
            if t is not None:
                succeeded += 1
            cell = f"{t:.6f}s" if t is not None else "FAIL"
            row += f"  {cell:>{col_width}s}"
        print(row, flush=True)

    print("-" * len(header))
    print(f"{succeeded}/{total} runs completed", flush=True)


if __name__ == "__main__":
    main()
