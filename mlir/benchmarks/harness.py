import argparse
import torch
import time
import docc.torch


def _torch_to_device(
    input: None | int | float | torch.Tensor | tuple, device: torch.device
) -> None | int | float | torch.Tensor | tuple:
    if type(input) in [None, int, float]:
        return input
    elif type(input) == torch.Tensor:
        return input.to(device)
    elif type(input) == tuple:
        res = []
        for elem in input:
            res.append(_torch_to_device(elem, device))
        return tuple(res)
    else:
        raise ValueError(f"Unknown type for copying to device: {type(input)}")


def _torch_from_device(
    output: None | int | float | torch.Tensor | tuple,
) -> None | int | float | torch.Tensor | tuple:
    if type(output) in [None, int, float]:
        return output
    elif type(output) == torch.Tensor:
        return output.to("cpu")
    elif type(output) == tuple:
        res = []
        for elem in output:
            res.append(_torch_from_device(elem))
        return tuple(res)
    else:
        raise ValueError(f"Unknown type for copying from device: {type(output)}")


def run_benchmark(setup_func, name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--docc", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument(
        "--target",
        type=str,
        choices=["none", "sequential", "openmp", "cuda", "rocm"],
        default="none",
    )
    parser.add_argument("--n_runs", type=int, default=10)
    args = parser.parse_args()

    model, model_input = setup_func()

    if args.torch:
        # Check for GPU and set device accordingly
        if args.target in ["cuda", "rocm"]:
            if not torch.cuda.is_available():
                print(f"Error: Target {args.target} not available")
                exit(1)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Move model to GPU if necessary
        model = model.to(device)

        # Print device
        if device.type == "cuda":
            print(f"Torch device: {torch.cuda.get_device_name(device)}")
        else:
            print(f"Torch device: {device}")

        for _ in range(args.n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            with torch.no_grad():
                if device.type == "cuda":
                    model_input = _torch_to_device(model_input, device)
                program = torch.compile(model)
                if type(model_input) == tuple:
                    model_output = program(*model_input)
                else:
                    model_output = program(model_input)
                if device.type == "cuda":
                    _torch_from_device(model_output)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()
            print(f"{name} torch execution time: {end - start:.6f} seconds")

    if args.docc:
        for _ in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(
                    model,
                    backend="docc",
                    options={"target": args.target, "category": "server"},
                )
                if type(model_input) == tuple:
                    program(*model_input)
                else:
                    program(model_input)
            end = time.time()
            print(f"{name} docc execution time: {end - start:.6f} seconds")
