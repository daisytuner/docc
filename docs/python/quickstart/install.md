---
title: Install docc for Python
sidebar_label: Install
sidebar_position: 1
description: Install the docc Python package and prepare your system
---

The docc Python package compiles Python functions that work on NumPy arrays into native code for CPUs and GPUs. You add a decorator; docc compiles the function the first time it is called. In this Quickstart, you install the package and then run the WAXPBY example on all CPU cores and on a GPU.

## Requirements

| Component         | Requirement                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| Python            | 3.11 – 3.14                                                                            |
| Operating system  | Linux (x86-64, arm64, glibc ≥ 2.28), macOS ≥ 14 (Apple silicon)                        |
| C++ compiler      | Linux: `clang++-21` from the [official LLVM repository](https://apt.llvm.org/); macOS: Xcode Command Line Tools |
| OpenMP on macOS   | `brew install libomp`                                                                  |
| NVIDIA GPUs       | CUDA toolkit in `/usr/local/cuda` (only for `target="cuda"`)                           |
| AMD GPUs          | ROCm in `/opt/rocm` or `$ROCM_PATH` (only for `target="rocm"`)                         |

docc generates C++ code and compiles it with Clang when your function is called for the first time. This is why a C++ compiler is required at runtime.

## Install

1. On Linux, install Clang 21:

   ```bash
   wget https://apt.llvm.org/llvm.sh
   chmod +x llvm.sh
   sudo ./llvm.sh 21
   ```

2. Install the [`docc-compiler`](https://pypi.org/project/docc-compiler/) package, preferably in a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install docc-compiler
   ```

3. Check the installation:

   ```bash
   python -c "from docc.python import native; print('docc is ready')"
   ```

## Prepare GPU targets (optional)

docc detects the first GPU in your system and compiles for its architecture. To select a different one, set:

```bash
export DOCC_CUDA_ARCH=sm_90     # NVIDIA H100
export DOCC_ROCM_ARCH=gfx942    # AMD MI300
```

## Next step

Continue with [Run Your First Code](./first-program.md) to compile and accelerate the WAXPBY example.
