import argparse
import torch
import time
import docc.torch

def run_benchmark(setup_func, name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--docc", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--target", type=str, default="none")
    parser.add_argument("--n_runs", type=int, default=10)
    args = parser.parse_args()

    model, model_input = setup_func()

    if args.torch:
        for _ in range(args.n_runs):
            start = time.time()
            program = torch.compile(model)
            program(model_input)
            end = time.time()
            print(f"Torch execution time: {end - start:.6f} seconds")
    
    if args.docc:
        docc.torch.set_backend_options(target=args.target, category="server")
        for _ in range(args.n_runs):
            start = time.time()
            program = torch.compile(model, backend="docc")
            program(model_input)
            end = time.time()
            print(f"Torch execution time: {end - start:.6f} seconds")
