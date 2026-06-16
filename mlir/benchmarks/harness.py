import argparse
import torch
import time
import docc.torch


def check_correctness(model, model_input, name, target, remote_tuning, rtol=1e-3, atol=1e-3):
    """Compare DOCC output against eager PyTorch reference."""
    with torch.no_grad():
        if type(model_input) == tuple:
            ref_output = model(*model_input)
        else:
            ref_output = model(model_input)

    with torch.no_grad():
        program = torch.compile(
            model,
            backend="docc",
            options={
                "target": target,
                "category": "server",
                "remote_tuning": remote_tuning,
            },
        )
        if type(model_input) == tuple:
            docc_output = program(*model_input)
        else:
            docc_output = program(model_input)

    # Move to CPU for comparison
    if isinstance(ref_output, torch.Tensor):
        ref_cpu = ref_output.cpu()
        docc_cpu = docc_output.cpu()
        if torch.allclose(ref_cpu, docc_cpu, rtol=rtol, atol=atol):
            max_diff = (ref_cpu - docc_cpu).abs().max().item()
            print(f"[CORRECTNESS] {name}: PASS (max abs diff: {max_diff:.2e})")
        else:
            max_diff = (ref_cpu - docc_cpu).abs().max().item()
            mean_diff = (ref_cpu - docc_cpu).abs().mean().item()
            print(
                f"[CORRECTNESS] {name}: FAIL "
                f"(max abs diff: {max_diff:.2e}, mean abs diff: {mean_diff:.2e}, "
                f"rtol={rtol}, atol={atol})"
            )
    elif isinstance(ref_output, (tuple, list)):
        all_pass = True
        for i, (r, d) in enumerate(zip(ref_output, docc_output)):
            r_cpu = r.cpu()
            d_cpu = d.cpu()
            if not torch.allclose(r_cpu, d_cpu, rtol=rtol, atol=atol):
                max_diff = (r_cpu - d_cpu).abs().max().item()
                print(f"[CORRECTNESS] {name} output[{i}]: FAIL (max abs diff: {max_diff:.2e})")
                all_pass = False
        if all_pass:
            print(f"[CORRECTNESS] {name}: PASS (all {len(ref_output)} outputs match)")
    else:
        print(f"[CORRECTNESS] {name}: SKIP (unsupported output type {type(ref_output)})")


def run_benchmark(setup_func, name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--docc", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--check", action="store_true", help="Run correctness check against eager PyTorch")
    parser.add_argument("--target", type=str, default="none")
    parser.add_argument("--remote_tuning", action="store_true")
    parser.add_argument("--n_runs", type=int, default=10)
    args = parser.parse_args()

    model, model_input = setup_func()

    if args.check:
        check_correctness(model, model_input, name, args.target, args.remote_tuning)
        return

    if args.torch:
        for _ in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(model)
                if type(model_input) == tuple:
                    program(*model_input)
                else:
                    program(model_input)
            end = time.time()
            print(f"{name} torch execution time: {end - start:.6f} seconds")

    if args.docc:
        for _ in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(
                    model,
                    backend="docc",
                    options={
                        "target": args.target,
                        "category": "server",
                        "remote_tuning": args.remote_tuning,
                    },
                )
                if type(model_input) == tuple:
                    program(*model_input)
                else:
                    program(model_input)
            end = time.time()
            print(
                f"{name} docc execution time: {end - start:.6f} seconds "
                f"(remote_tuning={args.remote_tuning})"
            )
