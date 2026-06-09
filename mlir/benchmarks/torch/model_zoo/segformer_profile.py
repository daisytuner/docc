import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import SegformerForSemanticSegmentation

import docc.torch


SEGFORMER_MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


def resolve_model_name(version: str, model: str | None) -> str:
    if model:
        return model
    return SEGFORMER_MODELS[version]


def _assert_cuda_arch_supported() -> None:
    capability = torch.cuda.get_device_capability()
    current_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if current_arch not in supported_arches:
        supported_str = " ".join(sorted(supported_arches))
        raise RuntimeError(
            "The active PyTorch CUDA build does not support this GPU architecture "
            f"({current_arch}). Supported architectures: {supported_str}. "
            "Install a compatible CUDA wheel (for RTX 50xx typically cu128+), "
            "or run with --device cpu."
        )


def setup_segformer(
    model_name: str,
    model_device: str,
    image_size: int,
    input_device: str | None = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    if input_device is None:
        input_device = model_device

    model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()
    if model_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        _assert_cuda_arch_supported()
        model = model.to("cuda")

    if input_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA input requested but not available")

    model_input = torch.randn(1, 3, image_size, image_size, device=input_device)
    return model, model_input


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _materialize_output(res: object) -> None:
    if isinstance(res, dict):
        _ = {k: v.cpu() if torch.is_tensor(v) else v for k, v in res.items()}
    elif hasattr(res, "logits") and torch.is_tensor(res.logits):
        _ = res.logits.cpu()


def _run_once(program: torch.nn.Module, model_input: torch.Tensor, model_dev: torch.device) -> None:
    current_input = model_input
    if current_input.device != model_dev:
        current_input = current_input.to(model_dev, non_blocking=True)

    res = program(pixel_values=current_input)
    _materialize_output(res)
    if model_dev.type == "cuda":
        torch.cuda.synchronize(model_dev)


def run_torch_profile(model: torch.nn.Module, model_input: torch.Tensor, n_runs: int, trace_prefix: str) -> None:
    model_dev = _model_device(model)
    with torch.no_grad():
        compile_start = time.perf_counter()
        program = torch.compile(model)
        _run_once(program, model_input, model_dev)
        compile_end = time.perf_counter()
        print(f"Torch compile+first-run: {(compile_end - compile_start):.6f} s")

        _run_once(program, model_input, model_dev)
        activities = [ProfilerActivity.CPU]
        if model_dev.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        for i in range(n_runs):
            start = time.perf_counter()
            with profile(activities=activities, record_shapes=True) as prof:
                _run_once(program, model_input, model_dev)
            end = time.perf_counter()

            trace_path = f"{trace_prefix}_torch_{i}.json"
            prof.export_chrome_trace(trace_path)
            print(f"Torch runtime run {i}: {(end - start):.6f} s, trace={trace_path}")


def run_docc_profile(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    n_runs: int,
    target: str,
    remote_tuning: bool,
    trace_prefix: str,
) -> None:
    model_dev = _model_device(model)
    with torch.no_grad():
        compile_start = time.perf_counter()
        program = torch.compile(
            model,
            backend="docc",
            options={"target": target, "category": "server", "remote_tuning": remote_tuning},
        )
        _run_once(program, model_input, model_dev)
        compile_end = time.perf_counter()
        print(
            f"DOCC compile+first-run ({target}, remote_tuning={remote_tuning}): "
            f"{(compile_end - compile_start):.6f} s"
        )

        _run_once(program, model_input, model_dev)
        activities = [ProfilerActivity.CPU]
        if model_dev.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        for i in range(n_runs):
            start = time.perf_counter()
            with profile(activities=activities, record_shapes=True) as prof:
                _run_once(program, model_input, model_dev)
            end = time.perf_counter()

            trace_path = f"{trace_prefix}_docc_{target}_{i}.json"
            prof.export_chrome_trace(trace_path)
            print(f"DOCC runtime run {i}: {(end - start):.6f} s, trace={trace_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SegFormer with Torch and/or DOCC backend")
    parser.add_argument("--docc", action="store_true", help="Run DOCC backend")
    parser.add_argument("--torch", action="store_true", dest="run_torch", help="Run Torch backend")
    parser.add_argument(
        "--version",
        type=str,
        choices=list(SEGFORMER_MODELS.keys()),
        default="b0",
        help="SegFormer variant to use when --model is not provided",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Hugging Face model id to override --version",
    )
    parser.add_argument("--target", type=str, default="none", help="DOCC target")
    parser.add_argument(
        "--remote_tuning",
        action="store_true",
        help="Enable DOCC remote tuning during compilation",
    )
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs per backend")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for model and input tensor",
    )
    parser.add_argument(
        "--input_device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device where input tensor is created (defaults to --device)",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument(
        "--trace_prefix",
        type=str,
        default="segformer_trace",
        help="Prefix for exported Torch profiler traces",
    )
    args = parser.parse_args()

    if not args.docc and not args.run_torch:
        parser.error("Specify at least one backend: --torch and/or --docc")

    return args


def main() -> None:
    args = parse_args()
    model_name = resolve_model_name(args.version, args.model)
    input_device = args.input_device if args.input_device is not None else args.device
    model, model_input = setup_segformer(
        model_name,
        args.device,
        args.image_size,
        input_device=input_device,
    )

    print(f"Model: {model_name}")
    print(f"Device: {args.device}")
    print(f"Input device: {input_device}")
    print(f"Remote tuning: {args.remote_tuning}")
    print(f"Runs: {args.n_runs}")

    if args.run_torch:
        run_torch_profile(model, model_input, args.n_runs, args.trace_prefix)

    if args.docc:
        run_docc_profile(
            model,
            model_input,
            args.n_runs,
            args.target,
            args.remote_tuning,
            args.trace_prefix,
        )


if __name__ == "__main__":
    main()
