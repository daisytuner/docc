import argparse
import torch
import time
import docc.torch

from torch.profiler import profile, schedule, ProfilerActivity

my_schedule = schedule(skip_first=10, wait=5, warmup=5, active=1, repeat=5)
RESULT_DIR = "./prof_trace"


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    # print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")


def run_benchmark(setup_func, name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--docc", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--torch-gpu", action="store_true")
    parser.add_argument("--torch-rocm", action="store_true")
    parser.add_argument("--target", type=str, default="none")
    parser.add_argument("--n_runs", type=int, default=31)
    args = parser.parse_args()

    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"

    model, model_input = setup_func()

    if args.torch:
        mini = 1000000000.0
        for i in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(model)
                program(model_input)
            end = time.time()
            print(f"{name} torch execution time: {end - start:.6f} seconds")
            if i != 0:
                mini = min(mini, end - start)
        print((f"{(mini*1000):.6f}").replace(".", ",") + " ms")

    if args.torch_gpu:
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        mini = 1000000000.0
        for i in range(args.n_runs):
            gpu_input = model_input.to(gpu)
            start = time.time()
            # with torch.no_grad():
            gpu_model = model.to(gpu)
            # program = torch.compile(gpu_model)
            gpu_model.train()
            gpu_model.requires_grad_(True)
            out = gpu_model.forward(gpu_input)
            torch.cuda.synchronize()
            end = time.time()
            gpu_out = out.to(cpu)
            print(f"{name} torch-gpu execution time: {end - start:.6f} seconds")
            if i != 0:
                mini = min(mini, end - start)
        print((f"{(mini*1000):.6f}").replace(".", ",") + " ms")

    if args.torch_rocm:
        cpu = torch.device("cpu")
        gpu = torch.device("cuda")
        mini = 1000000000.0
        for i in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(model)
                model_input.to(gpu)
                out = program(model_input)
                out.to(cpu)
            end = time.time()
            print(f"{name} torch-rocm execution time: {end - start:.6f} seconds")
            if i != 0:
                mini = min(mini, end - start)
        print((f"{(mini*1000):.6f}").replace(".", ",") + " ms")

    if args.docc:
        docc.torch.set_backend_options(target=args.target, category="server")
        mini = 1000000000.0
        for i in range(args.n_runs):
            start = time.time()
            with torch.no_grad():
                program = torch.compile(model, backend="docc")
                program(model_input)
            end = time.time()
            print(f"{name} docc execution time: {end - start:.6f} seconds")
            if i != 0:
                mini = min(mini, end - start)
        print((f"{(mini*1000):.6f}").replace(".", ",") + " ms")
