---
title: Install docc for PyTorch
sidebar_label: Install
sidebar_position: 1
description: Install the docc backend for torch.compile and prepare your system
---

docc plugs into PyTorch as a `torch.compile` backend. You keep your model as it is, add one import, and select `backend="docc"`. In this Quickstart, you install the backend and then run an image classifier (ResNet-18) on all CPU cores and on a GPU.

## Requirements

| Component        | Requirement                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| Python           | 3.11 – 3.14                                                                                                     |
| Operating system | Linux (x86-64, arm64), macOS ≥ 14 (Apple silicon, CPU only)                                                     |
| PyTorch          | 2.13 (the tested version)                                                                                       |
| C++ compiler     | Linux: `clang++-21` from the [official LLVM repository](https://apt.llvm.org/); macOS: Xcode Command Line Tools |
| NVIDIA GPUs      | CUDA 12 and [CuPy](https://cupy.dev/) (`cupy-cuda12x`)                                                          |
| AMD GPUs         | ROCm 7 and [CuPy](https://cupy.dev/) (`cupy-rocm-7-0`)                                                          |

The backend generates C++ code for your model and compiles it with Clang, which is why a C++ compiler is required at runtime.

## Install

1. On Linux, install Clang 21:

   ```bash
   wget https://apt.llvm.org/llvm.sh
   chmod +x llvm.sh
   sudo ./llvm.sh 21
   ```

2. Create a virtual environment and install PyTorch for your hardware:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

   # CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   # NVIDIA GPUs (use cu129 for Blackwell GPUs such as RTX 50xx and B200)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && pip install cupy-cuda12x
   # AMD GPUs
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 && pip install cupy-rocm-7-0
   ```

3. Install the docc compiler and the PyTorch backend:

   ```bash
   pip install docc-compiler docc-pytorch
   ```

4. Check the installation:

   ```bash
   python -c "import docc.pytorch; print('docc backend is ready')"
   ```

## Prepare GPU targets (optional)

docc detects the first GPU in your system and compiles for its architecture. To select a different one, set:

```bash
export DOCC_CUDA_ARCH=sm_90     # NVIDIA H100
export DOCC_ROCM_ARCH=gfx942    # AMD MI300
```

## Next step

Continue with [Run Your First Model](./first-program.md) to compile and accelerate ResNet-18.
