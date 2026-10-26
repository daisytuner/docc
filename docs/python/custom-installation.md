---
title: Custom Installation
sidebar_position: 2
description: Build the docc Python package from source, including debug builds
---

Build the Python package from source if you want the latest `main` branch, a debug build, or to work on docc itself. The package contains the complete compiler (built with CMake through [scikit-build-core](https://scikit-build-core.readthedocs.io/)), so the build takes a while.

## Requirements

- Python ≥ 3.11
- CMake ≥ 3.16, Ninja, and a C++20 compiler (Clang 21 recommended)
- System libraries: GMP, zstd, ISL, nlohmann-json, Boost Graph

On Ubuntu:

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 21
sudo apt-get install -y cmake ninja-build git \
    libgmp-dev libzstd-dev libisl-dev nlohmann-json3-dev libboost-graph-dev
```

On macOS:

```bash
brew install cmake ninja gmp isl nlohmann-json boost libomp
```

## Build

```bash
git clone https://github.com/daisytuner/docc.git
cd docc
git submodule update --init --recursive

python3 -m venv .venv
source .venv/bin/activate
pip install -e python/
```

`pip install -e` creates an editable install: changes to the Python sources take effect immediately; changes to C++ sources require running the command again.

Check your build with the WAXPBY example from the [Quickstart](./quickstart/first-program.md):

```bash
python waxpby.py
```

## Build options

Pass CMake options with `-C cmake.define.<OPTION>=<value>` (or via the `CMAKE_ARGS` environment variable):

| Option                          | Default                                     | Description                                                                         |
| ------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| `DOCC_CXX_COMPILER`             | `clang++-21` (Linux), `clang++` (macOS)     | C++ compiler that the package uses at runtime to compile generated code             |
| `DOCC_BUILD_TARGET_TENSTORRENT` | `ON`                                        | Build the Tenstorrent target                                                        |
| `DOCC_BUILD_TARGET_ET`          | `OFF`                                       | Build the ETSoC target                                                              |

For example, to use the Clang that ships with ROCm for generated code:

```bash
pip install -e python/ -C cmake.define.DOCC_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
```

The compiler can also be changed at runtime with `DOCC_BACKEND_COMPILER`, see [Usage](./usage.md#environment-variables).

## Debug builds

For development, build the C++ parts with debug information and keep the CMake build directory, so that rebuilds are incremental:

```bash
pip install --no-build-isolation -e python/ \
    -C cmake.build-type=Debug \
    -C build-dir=build-python
```

`--no-build-isolation` requires the build dependencies in your environment: `pip install scikit-build-core pybind11`.

[Debug Options](./debugging.md) explains how to run the tests and inspect what docc does with your functions.
