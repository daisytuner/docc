import os
import time
import threading
import statistics
import contextlib
import numpy as np
import torch
import torchvision.models as models
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader

import docc.torch

# Imagenette local index → ImageNet-1K class index mapping.
_IMAGENETTE_TO_IMAGENET = {
    0: 0,  # n01440764 → tench
    1: 217,  # n02102040 → English springer
    2: 482,  # n02979186 → cassette player
    3: 491,  # n03000684 → chain saw
    4: 497,  # n03028079 → church
    5: 566,  # n03394916 → French horn
    6: 569,  # n03417042 → garbage truck
    7: 571,  # n03425413 → gas pump
    8: 574,  # n03445777 → golf ball
    9: 701,  # n03888257 → parachute
}


def _imagenette_dataset(preprocess, batch_size):
    data_dir = os.environ.get("IMAGENETTE_DIR", "/tmp/imagenette")
    dataset = Imagenette(
        root=data_dir, split="val", size="320px", download=True, transform=preprocess
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return dataset, dataloader


def _compute_metrics(preds_ref, preds_docc, labels):
    accuracy_ref = (preds_ref == labels).mean()
    accuracy_docc = (preds_docc == labels).mean()
    agreement = (preds_ref == preds_docc).mean()
    return {
        "accuracy_ref": float(accuracy_ref),
        "accuracy_docc": float(accuracy_docc),
        "agreement": float(agreement),
    }


def _compute_logit_correlation(logits_ref, logits_docc):
    ref = logits_ref - logits_ref.mean(axis=1, keepdims=True)
    docc = logits_docc - logits_docc.mean(axis=1, keepdims=True)

    num = (ref * docc).sum(axis=1)
    denom = np.sqrt((ref**2).sum(axis=1) * (docc**2).sum(axis=1))
    correlations = num / np.clip(denom, a_min=1e-12, a_max=None)
    return float(correlations.mean())


def setup():
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    dataset, dataloader = _imagenette_dataset(preprocess, batch_size)

    # Collect all batches into a list of (images, labels) tuples
    batches = []
    for images, labels in dataloader:
        current_batch_size = images.shape[0]
        if current_batch_size < batch_size:
            padding = torch.zeros(
                (batch_size - current_batch_size, *images.shape[1:]), dtype=images.dtype
            )
            images = torch.cat([images, padding], dim=0)
        batches.append((images, labels))

    return model, batches


def run_accuracy_evaluation(target):
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    weights = models.ResNet18_Weights.IMAGENET1K_V1

    # Reference model (fp32)
    model_ref = models.resnet18(weights=weights)
    model_ref.eval()

    # DOCC-compiled model
    model = models.resnet18(weights=weights)
    model.eval()
    docc.torch.set_backend_options(target=target, category="server")
    program = torch.compile(model, backend="docc")

    preprocess = weights.transforms()
    dataset, dataloader = _imagenette_dataset(preprocess, batch_size)

    all_preds_ref = []
    all_preds_docc = []
    all_logits_ref = []
    all_logits_docc = []
    all_labels = []

    with torch.no_grad():
        total = len(dataloader)
        for i, (images, labels) in enumerate(dataloader, 1):
            print(f"\r  [{i}/{total}] Processing batch...", end="", flush=True)

            current_batch_size = images.shape[0]
            if current_batch_size < batch_size:
                padding = torch.zeros(
                    (batch_size - current_batch_size, *images.shape[1:]),
                    dtype=images.dtype,
                )
                images = torch.cat([images, padding], dim=0)

            ref_out = model_ref(images)
            docc_out = program(images)

            if current_batch_size < batch_size:
                ref_out = ref_out[:current_batch_size]
                docc_out = docc_out[:current_batch_size]

            all_preds_ref.append(ref_out.argmax(dim=1))
            all_preds_docc.append(docc_out.argmax(dim=1))
            all_logits_ref.append(ref_out)
            all_logits_docc.append(docc_out)

            mapped = torch.tensor([_IMAGENETTE_TO_IMAGENET[l.item()] for l in labels])
            all_labels.append(mapped)
        print()

    preds_ref = torch.cat(all_preds_ref).numpy()
    preds_docc = torch.cat(all_preds_docc).numpy()
    logits_ref = torch.cat(all_logits_ref).numpy()
    logits_docc = torch.cat(all_logits_docc).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = _compute_metrics(preds_ref, preds_docc, labels)
    logit_corr = _compute_logit_correlation(logits_ref, logits_docc)

    print(f"\n{'='*60}")
    print("ResNet18 ImageNet Classification Correlation Report")
    print(f"{'='*60}")
    print(f"Dataset           : Imagenette validation ({len(dataset)} samples)")
    print(f"Reference (fp32)  : Top-1 accuracy = {metrics['accuracy_ref']:.4f}")
    print(f"DOCC      (bf16)  : Top-1 accuracy = {metrics['accuracy_docc']:.4f}")
    print(f"Agreement rate    : {metrics['agreement']:.4f}")
    print(f"Logit correlation : {logit_corr:.4f}")
    print(f"{'='*60}")

    assert (
        metrics["agreement"] > 0.85
    ), f"Agreement rate {metrics['agreement']:.4f} is below threshold 0.85"

    assert (
        logit_corr > 0.90
    ), f"Mean logit correlation {logit_corr:.4f} is below threshold 0.90"

    assert metrics["accuracy_docc"] > metrics["accuracy_ref"] - 0.15, (
        f"DOCC accuracy {metrics['accuracy_docc']:.4f} dropped too far "
        f"below reference {metrics['accuracy_ref']:.4f}"
    )


class EnergyMeasurement:
    """Context manager that measures CPU (RAPL) and GPU (NVML/amdsmi) energy."""

    _RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
    _RAPL_CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"

    def __init__(self, gpu_index=0, sample_interval=0.01, gpu_backend=None):
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self._requested_backend = gpu_backend  # None=auto, "nvml", or "amdsmi"
        self.cpu_energy_j = 0.0
        self.cpu_core_energy_j = 0.0
        self.gpu_energy_j = 0.0
        self._gpu_available = False
        self._gpu_backend = None  # "nvml" or "amdsmi"
        self._cpu_available = False

    def _read_rapl(self, path):
        with open(path) as f:
            return int(f.read())

    def _gpu_sampler_nvml(self):
        import pynvml

        while self._sampling:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                self._gpu_samples.append(mw)  # milliwatts
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def _gpu_sampler_amdsmi(self):
        import amdsmi

        while self._sampling:
            try:
                info = amdsmi.amdsmi_get_power_info(self._gpu_handle)
                watts = info["average_socket_power"]
                if isinstance(watts, (int, float)):
                    self._gpu_samples.append(watts * 1000.0)  # W -> mW
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def _init_nvml(self):
        import pynvml

        pynvml.nvmlInit()
        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)  # test read
        self._gpu_backend = "nvml"

    def _init_amdsmi(self):
        import amdsmi

        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        if self.gpu_index >= len(devices):
            raise RuntimeError(
                f"AMD GPU index {self.gpu_index} not found ({len(devices)} devices)"
            )
        self._gpu_handle = devices[self.gpu_index]
        info = amdsmi.amdsmi_get_power_info(self._gpu_handle)
        if not isinstance(info["average_socket_power"], (int, float)):
            raise RuntimeError("amdsmi power reading not available")
        self._gpu_backend = "amdsmi"

    def _init_gpu(self):
        if self._requested_backend == "amdsmi":
            self._init_amdsmi()
            return True
        if self._requested_backend == "nvml":
            self._init_nvml()
            return True

        # Auto-detect: try NVML (NVIDIA) first, then amdsmi (AMD)
        try:
            self._init_nvml()
            return True
        except Exception:
            pass
        try:
            self._init_amdsmi()
            return True
        except Exception:
            pass

        return False

    def __enter__(self):
        # CPU RAPL
        try:
            self._cpu_start = self._read_rapl(self._RAPL_PKG)
            self._cpu_core_start = self._read_rapl(self._RAPL_CORE)
            self._cpu_available = True
        except (PermissionError, FileNotFoundError):
            self._cpu_available = False

        # GPU
        self._gpu_samples = []
        self._sampling = True
        if self._init_gpu():
            sampler = (
                self._gpu_sampler_nvml
                if self._gpu_backend == "nvml"
                else self._gpu_sampler_amdsmi
            )
            self._gpu_thread = threading.Thread(target=sampler, daemon=True)
            self._gpu_thread.start()
            self._gpu_available = True

        return self

    def __exit__(self, *args):
        if self._cpu_available:
            cpu_end = self._read_rapl(self._RAPL_PKG)
            cpu_core_end = self._read_rapl(self._RAPL_CORE)
            self.cpu_energy_j = (cpu_end - self._cpu_start) * 1e-6
            self.cpu_core_energy_j = (cpu_core_end - self._cpu_core_start) * 1e-6

        if self._gpu_available:
            self._sampling = False
            self._gpu_thread.join()
            # Integrate power samples: sum(mW) * interval_s * 1e-3 = J
            self.gpu_energy_j = sum(self._gpu_samples) * self.sample_interval * 1e-3

    def report(self):
        parts = []
        if self._cpu_available:
            parts.append(
                f"CPU pkg={self.cpu_energy_j:.2f}J core={self.cpu_core_energy_j:.2f}J"
            )
        if self._gpu_available:
            backend = self._gpu_backend
            parts.append(
                f"GPU[{backend}]={self.gpu_energy_j:.2f}J ({len(self._gpu_samples)} samples)"
            )
        return "  ".join(parts) if parts else "no energy data"


BENCHMARKS = {
    "default": setup,
}


def run_benchmark(setup_func, name):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--docc", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--torch-gpu", action="store_true")
    parser.add_argument("--torch-rocm", action="store_true")
    parser.add_argument("--target", type=str, default="none")
    parser.add_argument("--n_runs", type=int, default=31)
    parser.add_argument("--energy", action="store_true", help="Measure CPU+GPU energy")
    bm_args = parser.parse_args()

    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"

    model, batches = setup_func()

    def run_all_batches(program, device=None):
        for images, _ in batches:
            if device is not None:
                images = images.to(device)
            program(images)

    def _gpu_backend_for_mode(mode_name):
        if mode_name == "docc":
            if "rocm" in bm_args.target:
                return "amdsmi"
            if "cuda" in bm_args.target:
                return "nvml"
        if mode_name == "torch-rocm":
            return "amdsmi"
        if mode_name == "torch-gpu":
            return "nvml"
        return None  # auto-detect

    def _run_mode(mode_name, compile_fn, device=None):
        times = []
        energies = []
        for i in range(bm_args.n_runs):
            gpu_be = _gpu_backend_for_mode(mode_name) if bm_args.energy else None
            ctx = (
                EnergyMeasurement(gpu_backend=gpu_be)
                if bm_args.energy
                else contextlib.nullcontext()
            )
            with ctx as em:
                start = time.time()
                with torch.no_grad():
                    program = compile_fn()
                    run_all_batches(program, device=device)
                end = time.time()
            elapsed = end - start
            print(f"{name} {mode_name} execution time: {elapsed:.6f} seconds", end="")
            if bm_args.energy and em is not None:
                print(f"  | {em.report()}", end="")
                if i != 0:
                    energies.append(em)
            print()
            if i != 0:
                times.append(elapsed)
        median = statistics.median(times)
        print((f"{(median*1000):.6f}").replace(".", ",") + " ms")
        if bm_args.energy and energies:
            cpu_median = statistics.median([e.cpu_energy_j for e in energies])
            gpu_median = statistics.median([e.gpu_energy_j for e in energies])
            print(
                f"Energy median: CPU pkg={cpu_median:.2f}J  GPU={gpu_median:.2f}J  total={cpu_median+gpu_median:.2f}J"
            )

    if bm_args.torch:
        _run_mode("torch", lambda: torch.compile(model))

    if bm_args.torch_gpu:
        device = torch.device("cuda")
        _run_mode("torch-gpu", lambda: torch.compile(model.to(device)), device=device)

    if bm_args.torch_rocm:
        device = torch.device("cuda")  # ROCm uses 'cuda' device via HIP
        _run_mode("torch-rocm", lambda: torch.compile(model.to(device)), device=device)

    if bm_args.docc:
        print(f"Running DOCC benchmark with target={bm_args.target}...")
        docc.torch.set_backend_options(target=bm_args.target, category="server")
        _run_mode("docc", lambda: torch.compile(model, backend="docc"))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="imagenet on resnet18 benchmark")
    parser.add_argument(
        "--variant", type=str, choices=list(BENCHMARKS.keys()), default="default"
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run full-dataset accuracy evaluation against docc",
    )
    args, remaining = parser.parse_known_args()

    # torch.set_float32_matmul_precision("highest")

    if args.accuracy:
        # Parse --target only for accuracy mode
        acc_parser = argparse.ArgumentParser()
        acc_parser.add_argument("--target", type=str, default="none")
        acc_args, _ = acc_parser.parse_known_args(remaining)
        run_accuracy_evaluation(acc_args.target)
    else:
        sys.argv = [sys.argv[0]] + remaining
        run_benchmark(BENCHMARKS[args.variant], f"imagenet_on_resnet18 {args.variant}")
