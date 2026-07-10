import argparse
import os
import sys
import torch
import time
import numpy as np
import pytest

import docc.torch

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
    parser.add_argument("--n_runs", type=int, default=10)
    args = parser.parse_args()

    # ROCm uses the same CUDA API in PyTorch
    if args.device in ("rocm", "cuda"):
        if not torch.cuda.is_available():
            print(f"{args.device.upper()} not available, exiting.", file=sys.stderr)
            sys.exit(1)
        device = torch.device("cuda")
    elif args.device == "docc":
        if args.target == "cuda" or args.target == "rocm":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    def sync_fn():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    compile_kwargs: dict = {}
    backend_label = args.device
    if args.device == "docc":
        compile_kwargs["backend"] = "docc"
        compile_kwargs["options"] = {
            "target": args.target,
            "category": "server",
            # "remote_tuning": True,
        }

        backend_label = f"docc_{args.target}"

    print(f"Backend: {backend_label}  |  torch device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

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
