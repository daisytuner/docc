import argparse
import os
import sys
import torch
import torch._dynamo
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

    device = torch.device("cpu")
    # Select device for torch. DOCC backend checks after device_residency
    if args.device in ("rocm", "cuda"):
        if not torch.cuda.is_available():
            print(f"{args.device.upper()} not available, exiting.", file=sys.stderr)
            sys.exit(1)
        device = torch.device("cuda")

    def sync_fn():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    backend_label = args.device
    compile_kwargs: dict = {}
    residency: dict = {"device_resident": None}
    if args.device == "docc":

        def _record_residency(info):
            # fullgraph inference compiles a single graph; record its residency.
            residency["device_resident"] = info["device_resident"]

        compile_kwargs["backend"] = "docc"
        compile_kwargs["options"] = {
            "target": args.target,
            "category": "server",
            "remote_tuning": args.remote_tuning,
            "on_compile": _record_residency,
        }

        backend_label = f"docc_{args.target}"

    model, model_input = setup_func(batch_size)
    model = model.eval()
    model.to(device)
    model.requires_grad_(False)
    sync_fn()
    program = torch.compile(model, fullgraph=True, **compile_kwargs)

    # Warmup: the first invocation absorbs one-time cold-start costs (TorchDynamo
    # tracing, docc compilation, CUDA context init). Run it untimed and outside
    # the perf-counted region so measurements reflect the steady-state runtime.
    #
    # For the docc backend this first, host-only run doubles as a probe: it builds
    # the artifact and reports through on_compile whether it ended up
    # device-resident. Host (cpu) inputs are accepted by both host and
    # device-resident artifacts, so this never raises regardless of the outcome.
    x = _prepare_input(model_input, device)
    x = _detach_input(x)
    with torch.no_grad():
        _invoke(program, x)

    sync_fn()

    # Residency is now known. A device-resident artifact consumes device pointers
    # zero-copy, so move model + inputs to the GPU. Changing the model's device
    # invalidates dynamo's guards, so reset and recompile, then warm up again to
    # keep the (full) GPU rebuild out of the measured region.
    if args.device == "docc" and residency.get("device_resident"):
        print(f"{name} {backend_label}: device-resident artifact", flush=True)
        device = torch.device("cuda")
        model.to(device)
        torch._dynamo.reset()
        program = torch.compile(model, fullgraph=True, **compile_kwargs)
        x = _prepare_input(model_input, device)
        x = _detach_input(x)
        with torch.no_grad():
            _invoke(program, x)

    sync_fn()

    print(f"Backend: {backend_label}  |  torch device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    # Start the measurements
    perf = PerfControl.from_env()
    reset_instrumentation()

    with perf.measure():
        for i in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                out = _invoke(program, x)
            sync_fn()
            end = time.time()
            print(f"{name} {backend_label} execution time: {end - start:.6f} seconds")


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
