---
title: Custom Installation
sidebar_position: 2
description: Build the docc PyTorch backend and the compiler from source, including debug builds
---

Build the backend from source if you want the latest `main` branch, a debug build, or to work on docc itself. The backend (`pytorch/`) is pure Python and builds on the docc Python package (`python/`), which contains the compiler. Both are installed from the same checkout so that their versions match.

## Requirements

The requirements of the [Python Custom Installation](../python/custom-installation.md#requirements) apply: Clang 21, CMake, Ninja, and the system libraries.

:::tip
Skip the setup with the [Docker images](../docker.md): `daisytuner/docc-build-env-llvm21-ubuntu-24.04` contains all build dependencies, and the run-env images add CUDA and ROCm. PyTorch itself is installed from the requirements files below.
:::

## Build

```bash
git clone https://github.com/daisytuner/docc.git
cd docc
git submodule update --init --recursive

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

1. Install PyTorch and the build dependencies. Choose the requirements file for your hardware:

   | File                                | Installs                                                      |
   | ----------------------------------- | ------------------------------------------------------------- |
   | `pytorch/requirements.txt`          | PyTorch for CPUs                                              |
   | `pytorch/requirements-cuda.txt`     | PyTorch for CUDA 12.6 and CuPy                                |
   | `pytorch/requirements-rocm.txt`     | PyTorch for ROCm 7.2 and CuPy                                 |
   | `pytorch/requirements-base.txt`     | Only the build and test dependencies; bring your own PyTorch  |

   ```bash
   pip install -r pytorch/requirements.txt
   ```

   The CUDA 12.6 build of PyTorch does not support Blackwell GPUs (RTX 50xx, B200). For those, install `requirements-base.txt`, `cupy-cuda12x`, and PyTorch from the `cu129` index instead.

2. Build and install the compiler:

   ```bash
   pip install --no-build-isolation -e python/
   ```

3. Install the backend:

   ```bash
   pip install --no-build-isolation -e pytorch/
   ```

Check your build with the ResNet-18 example from the [Quickstart](./quickstart/first-program.md):

```bash
python pytorch/examples/resnet18.py
```

Changes to the Python sources of both packages take effect immediately. After changing C++ sources, run step 2 again.

## Build options

The compiler build accepts the same CMake options as the Python package, for example the C++ compiler used for generated code. See [Python Custom Installation](../python/custom-installation.md#build-options).

## Debug builds

Build the compiler with debug information and keep the CMake build directory, so that rebuilds after C++ changes are incremental:

```bash
pip install --no-build-isolation -e python/ \
    -C cmake.build-type=Debug \
    -C build-dir=build-python
```

[Debug Options](./debugging.md) explains how to run the tests and inspect what docc does with your model.
