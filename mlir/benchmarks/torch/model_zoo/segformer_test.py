import time

import torch
from transformers import SegformerForSemanticSegmentation

import pytest

from torch_mlir import fx

import docc.torch

import os
os.environ["DOCC_STATISTICS"] = "1"
os.environ["DOCC_PROFILE_COMPILE"] = "1"
os.environ["DOCC_DEBUG"] = "dump"


def test_backend():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).eval()
    model_ref = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).eval()
    model_ref.load_state_dict(model.state_dict())

    example_input = torch.randn(1, 3, 512, 512)

    start = time.perf_counter()
    program = torch.compile(model, backend="docc", options={"target": "none", "category": "server"})
    end = time.perf_counter()
    print(f"compilation time: {(end - start) * 1000:.2f} ms")

    start = time.perf_counter()
    ref_program = torch.compile(model, backend="docc", options={"target": "none", "category": "server"})
    end = time.perf_counter()
    print(f"ref compilation time: {(end - start) * 1000:.2f} ms")

    with torch.no_grad():
        start = time.perf_counter()
        res = program(pixel_values=example_input)
        end = time.perf_counter()
        print(f"inference time: {(end - start) * 1000:.2f} ms")
        start = time.perf_counter()
        res_ref = ref_program(pixel_values=example_input)
        end = time.perf_counter()
        print(f"reference inference time: {(end - start) * 1000:.2f} ms")
        for k in range(res.logits.shape[0]):
            diff = (res.logits[k] - res_ref.logits[k]).abs()
            rel = (diff / res_ref.logits[k].abs().clamp(min=1e-8))
            n_total = diff.numel()
            n_fail = (~torch.isclose(res.logits[k], res_ref.logits[k], rtol=1e-2, atol=1e-4)).sum().item()
            print(
                f"Logits[{k}]: "
                f"abs_diff max={diff.max().item():.6f} mean={diff.mean().item():.6f} "
                f"| rel_err max={rel.max().item():.6f} mean={rel.mean().item():.6f} "
                f"| failing={n_fail}/{n_total} ({100*n_fail/n_total:.2f}%)"
            )
    assert torch.allclose(res.logits, res_ref.logits, rtol=1e-2, atol=1e-4)

@pytest.mark.skip("Skip")
def test_compile():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).eval()
    model_ref = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).eval()
    model_ref.load_state_dict(model.state_dict())

    example_input = torch.randn(1, 3, 512, 512)

    program = docc.torch.compile_torch(model, example_input, options={"target": "none", "category": "server"})
    with torch.no_grad():
        start = time.perf_counter()
        res = program(example_input)
        end = time.perf_counter()
        print(f"inference time: {(end - start) * 1000:.2f} ms")
        start = time.perf_counter()
        res_ref = model_ref(pixel_values=example_input)
        end = time.perf_counter()
        print(f"reference inference time: {(end - start) * 1000:.2f} ms")
    assert torch.allclose(res, res_ref.logits, rtol=1e-4)

def find_used_dialects():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).eval()

    example_input = torch.randn(1, 3, 512, 512)

    import re

    torch_mlir = fx.export_and_import(
        model, example_input, output_type=fx.OutputType.LINALG_ON_TENSORS
    )
    mlir_str = str(torch_mlir)
    mlir_str = re.sub(r"dense<[^>]+>", "dense<...>", mlir_str)
    mlir_str = re.sub(r': "0x[0-9A-Fa-f]+"', ': "..."', mlir_str)

    # Extract unique dialect.operation pairs
    ops = set(re.findall(r'\b([a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*)\b', mlir_str))
    print("Dialects and operations:")
    for op in sorted(ops):
        print(f"  {op}")
    print()

    # print(mlir_str)

def benchmark_segformer(model_name):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name
    ).eval()

    example_input = torch.randn(1, 3, 1024, 1024)

    program = torch.compile(model)
    with torch.no_grad():
        # Warmup
        res = program(pixel_values=example_input)

        import time
        import math
        from scipy import stats as scipy_stats

        times = []
        min_samples = 5
        max_samples = 500
        target_rel_ci = 0.01  # stop when 95% CI half-width < 1% of mean

        while len(times) < max_samples:
            start = time.perf_counter()
            res = program(pixel_values=example_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)

            if len(times) < min_samples:
                continue

            n = len(times)
            mean = sum(times) / n
            sem = scipy_stats.sem(times)
            half_width = scipy_stats.t.ppf(0.975, df=n - 1) * sem

            if half_width / mean < target_rel_ci:
                break

    n = len(times)
    mean = sum(times) / n
    sem = scipy_stats.sem(times)
    half_width = scipy_stats.t.ppf(0.975, df=n - 1) * sem
    print(f"Benchmarking {model_name}:")
    print(f"Average inference time: {mean:.2f} ms (n={n})")
    print(f"95% CI: [{mean - half_width:.2f}, {mean + half_width:.2f}] ms  (±{half_width:.2f} ms)")

if __name__ == "__main__":
    # find_used_dialects()
    find_used_dialects()
    #benchmark_segformer("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    #benchmark_segformer("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    #benchmark_segformer("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    #benchmark_segformer("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
    #benchmark_segformer("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")