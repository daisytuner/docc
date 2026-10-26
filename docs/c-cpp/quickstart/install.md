---
title: Install docc for C/C++
sidebar_label: Install
sidebar_position: 1
description: Install the prebuilt docc compiler for C/C++ and prepare your system
---

`docc` is a drop-in replacement for `clang` and `gcc`. In this Quickstart, you install it, compile a small vector kernel, and then run that kernel on all CPU cores and on a GPU without changing the code.

## Requirements

| Component          | Requirement                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| Operating system   | Ubuntu 24.04 (x86-64, arm64), Ubuntu 26.04, Debian 13, RHEL 10 (x86-64) |
| LLVM/Clang         | 21, from the [official LLVM repository](https://apt.llvm.org/)           |
| NVIDIA GPUs        | CUDA toolkit in `/usr/local/cuda` (only for `-docc-tune=cuda`)           |
| AMD GPUs           | ROCm in `/opt/rocm` or `$ROCM_PATH` (only for `-docc-tune=rocm`)         |
| Tenstorrent        | TT-Metalium, located via `$TT_METAL_INSTALL_DIR` or `$TT_METAL_HOME` (only for `-docc-tune=tenstorrent`) |

## Install

1. Install LLVM 21 from the official repository so that it does not conflict with your distribution's default LLVM:

   ```bash
   wget https://apt.llvm.org/llvm.sh
   chmod +x llvm.sh
   sudo ./llvm.sh 21
   ```

2. Download the package for your distribution from the [Daisytuner website](https://daisytuner.com/) and install it:

   ```bash
   # Debian / Ubuntu
   sudo apt-get install ./docc_*.deb

   # RHEL
   sudo dnf install ./docc-*.rpm
   ```

3. Check the installation:

   ```bash
   docc --version
   ```

The package installs the compiler drivers `docc` (C, like `clang`) and `docc-cpp` (C++, like `clang++`), together with the linker wrapper `docc-ld` and the runtime libraries.

## Prepare GPU targets (optional)

docc detects the first GPU in your system and compiles for its architecture. To compile for a different GPU, for example on a build machine without a GPU, set the architecture explicitly:

```bash
export DOCC_CUDA_ARCH=sm_90     # NVIDIA H100
export DOCC_ROCM_ARCH=gfx942    # AMD MI300
```

## Next step

Continue with [Run Your First Code](./first-program.md) to compile and accelerate the running example of this guide.
