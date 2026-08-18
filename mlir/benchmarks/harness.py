import argparse
from collections import deque
import os
import sys
import torch
import time
import numpy as np
import pytest

import docc.torch
from docc.benchmarks.perf import PerfControl
from docc.benchmarks import reset_instrumentation

os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 for CUDA and ROCm backends


def _prepare_input(model_input, device):
    """Move input(s) to the given device, recursing into nested tuples/lists."""
    if isinstance(model_input, torch.Tensor):
        return model_input.to(device)
    if isinstance(model_input, tuple):
        return tuple(_prepare_input(x, device) for x in model_input)
    if isinstance(model_input, list):
        return [_prepare_input(x, device) for x in model_input]
    return model_input


def _detach_input(model_input):
    """Detach input(s), recursing into nested tuples/lists."""
    if isinstance(model_input, torch.Tensor):
        return model_input.detach()
    if isinstance(model_input, tuple):
        return tuple(_detach_input(x) for x in model_input)
    if isinstance(model_input, list):
        return [_detach_input(x) for x in model_input]
    return model_input


def _invoke(model, model_input):
    """Call model, unpacking tuple inputs."""
    if isinstance(model_input, tuple):
        return model(*model_input)
    return model(model_input)


def _find_docc_device_resident(program):
    """Best-effort lookup for the Docc TorchProgram hidden by torch.compile."""
    seen = set()
    queue = deque([program])

    while queue and len(seen) < 2000:
        obj = queue.popleft()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        compiled = getattr(obj, "_compiled", None)
        if compiled is not None and hasattr(compiled, "device_resident"):
            return bool(compiled.device_resident)

        if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
            continue

        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    queue.append(cell.cell_contents)
                except ValueError:
                    pass

        obj_dict = getattr(obj, "__dict__", None)
        if obj_dict:
            queue.extend(obj_dict.values())

        if isinstance(obj, dict):
            queue.extend(obj.values())
        elif isinstance(obj, (tuple, list, set, frozenset)):
            queue.extend(obj)

    return None


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(setup_func, name, batch_size=32):
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--remote-tuning",
        action="store_true",
        help="Enable remote tuning for the Docc backend (only used when --device docc)",
    )
    parser.add_argument("--n_runs", type=int, default=10)
    args = parser.parse_args()

    # ROCm uses the same CUDA API in PyTorch
    if args.device in ("rocm", "cuda"):
        if not torch.cuda.is_available():
            print(f"{args.device.upper()} not available, exiting.", file=sys.stderr)
            sys.exit(1)
        requested_device = torch.device("cuda")
    elif args.device == "docc":
        if args.target in ("cuda", "rocm") and torch.cuda.is_available():
            requested_device = torch.device("cuda")
        else:
            requested_device = torch.device("cpu")
    else:
        requested_device = torch.device("cpu")

    device = torch.device("cpu") if args.device == "docc" else requested_device

    # torch-mlir cannot legalize the vendor-specific ops PyTorch emits on GPU
    # (e.g. aten.miopen_batch_norm on ROCm, which CUDA avoids by emitting the
    # torch-mlir-supported aten.cudnn_batch_norm). Disabling the cuDNN/MIOpen
    # backend forces batchnorm/convolution to decompose into portable native
    # aten ops that torch-mlir can lower, while the tensors stay resident on the
    # GPU so no host<->device transfer enters the timed region. This only
    # affects the traced representation used for export -- docc regenerates the
    # kernels for its target, so measured performance is unaffected.
    if args.device == "docc" and requested_device.type == "cuda":
        torch.backends.cudnn.enabled = False

    def sync_fn():
        _sync_device(device)

    compile_kwargs: dict = {}
    backend_label = args.device
    if args.device == "docc":
        compile_kwargs["backend"] = "docc"
        compile_kwargs["options"] = {
            "target": args.target,
            "category": "server",
            "remote_tuning": args.remote_tuning,
        }

        backend_label = f"docc_{args.target}"

    print(f"Backend: {backend_label}  |  torch device: {device}", flush=True)
    if requested_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(requested_device)}", flush=True)

    model, model_input = setup_func(batch_size)
    model = model.eval()
    program = torch.compile(model, fullgraph=True, **compile_kwargs)

    start = time.time()
    model.to(device)
    model.requires_grad_(False)
    x = _prepare_input(model_input, device)
    x = _detach_input(x)
    sync_fn()
    end = time.time()
    print(f"{name} {backend_label} setup time: {end - start:.6f} seconds")

    perf = PerfControl.from_env()

    # Warmup: the first invocation absorbs one-time cold-start costs (TorchDynamo
    # tracing, docc reuse-binary load / SDFG parse, CUDA context init). Run it
    # untimed and outside the perf-counted region so measurements reflect the
    # steady-state runtime.
    with torch.no_grad():
        _invoke(program, x)
    sync_fn()

    if args.device == "docc" and requested_device.type == "cuda":
        device_resident = _find_docc_device_resident(program)
        print(f"Docc device resident: {device_resident}", flush=True)
        if device_resident:
            device = requested_device
            model.to(device)
            x = _prepare_input(model_input, device)
            x = _detach_input(x)
            with torch.no_grad():
                _invoke(program, x)
            sync_fn()

    # The RTL aggregates every region invocation, including the warmup above;
    # drop those so region counts/durations match only the measured runs. The
    # compiled artifact is hidden inside torch.compile, so reset all artifacts.
    reset_instrumentation()

    with perf.measure():
        for i in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                out = _invoke(program, x)
            sync_fn()
            end = time.time()
            print(f"{name} {backend_label} execution time: {end - start:.6f} seconds")

    start = time.time()
    if isinstance(out, torch.Tensor):
        out.to("cpu").detach()
        sync_fn()
    end = time.time()
    print(f"{name} {backend_label} output transfer time: {end - start:.6f} seconds")


def run_pytest(setup_func, target="none"):
    if sys.platform == "darwin":
        if target in ("cuda", "rocm"):
            return

    device = torch.device("cpu")

    model, model_input = setup_func()
    model = model.eval().to(device)

    # Run reference (plain eager execution, no compile)
    with torch.no_grad():
        x_ref = _prepare_input(model_input, device)
        out_ref = _invoke(model, x_ref)

    compiled_model = torch.compile(model, backend="docc", options={"target": target})

    with torch.no_grad():
        x_test = _prepare_input(model_input, device)
        out_test = _invoke(compiled_model, x_test)

    # Compare outputs
    if isinstance(out_ref, torch.Tensor):
        np.testing.assert_allclose(
            out_test.cpu().numpy(), out_ref.cpu().numpy(), rtol=1e-2, atol=1e-6
        )
    elif isinstance(out_ref, (tuple, list)):
        for ref, test in zip(out_ref, out_test):
            if isinstance(ref, torch.Tensor):
                np.testing.assert_allclose(
                    test.cpu().numpy(), ref.cpu().numpy(), rtol=1e-2, atol=1e-6
                )
    elif isinstance(out_ref, dict):
        for key in out_ref:
            if isinstance(out_ref[key], torch.Tensor):
                np.testing.assert_allclose(
                    out_test[key].cpu().numpy(),
                    out_ref[key].cpu().numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                )

    # For sequential: verify instrumented binary was produced and run it
    if target == "sequential":
        # Run a second time to exercise the instrumented/cached path
        with torch.no_grad():
            x_test2 = _prepare_input(model_input, device)
            out_test2 = _invoke(compiled_model, x_test2)

        if isinstance(out_ref, torch.Tensor):
            np.testing.assert_allclose(
                out_test2.cpu().numpy(), out_ref.cpu().numpy(), rtol=1e-2, atol=1e-6
            )
