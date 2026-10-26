---
title: Custom Installation
sidebar_position: 2
description: Build docc for C/C++ from source, configure CMake options, and create debug builds
---

Build docc from source if you want the latest `main` branch, a debug build, or to work on the compiler itself. At the end of this page, you compile the WAXPBY example with your own build.

## Requirements

- Linux (Ubuntu 24.04 is used in CI)
- CMake ≥ 3.16 and Ninja
- LLVM/Clang 21.1, including the development packages
- System libraries: GMP, zstd, ISL, nlohmann-json, Boost Graph (other Boost components are fetched automatically)

On Ubuntu:

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 21
sudo apt-get install -y cmake ninja-build git \
    llvm-21-dev libclang-21-dev libomp-21-dev \
    libgmp-dev libzstd-dev libisl-dev nlohmann-json3-dev libboost-graph-dev
```

:::tip
The CI build environments are published as Docker images, for example `daisytuner/docc-build-env-llvm21-ubuntu-24.04`. They contain all dependencies.
:::

## Build

```bash
git clone https://github.com/daisytuner/docc.git
cd docc
git submodule update --init --recursive

cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang-21 \
    -DCMAKE_CXX_COMPILER=clang++-21 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_FRONTEND=ON \
    -DLLVM_BUILD_TESTS=OFF \
    -DSDFG_BUILD_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_BENCHMARKS_GOOGLE=OFF
ninja -C build
```

## Use your build

The driver in the build tree finds its plugin and runtime libraries automatically. Compile the running example with it:

```bash
./build/llvm/driver/docc -O3 -docc-tune=openmp waxpby.c -o waxpby
```

To install your build system-wide, create a package and install it:

```bash
cd build
cpack -G DEB
sudo apt-get install ./docc-llvm*.deb
```

## CMake options

| Option                          | Default   | Description                                                                     |
| ------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `LLVM_BUILD_FRONTEND`           | `OFF`     | Build the C/C++ frontend (`docc`, `docc-cpp`, `docc-ld`)                         |
| `PYTHON_BUILD_FRONTEND`         | `OFF`     | Build the Python frontend, see [Python Custom Installation](../python/custom-installation.md) |
| `DOCC_BUILD_TARGET_TENSTORRENT` | `ON`      | Build the Tenstorrent target                                                    |
| `LLVM_BUILD_TESTS`              | `ON`      | Build the tests of the C/C++ frontend                                           |
| `SDFG_BUILD_TESTS`              | `ON`      | Build the tests of the core libraries                                           |
| `SDFG_ENABLE_SANITIZER`         | `OFF`     | Build with sanitizers                                                           |
| `SDFG_SANITIZER`                | `address` | Sanitizers to use: `address`, `memory`, `undefined`, `leak`, `thread`           |
| `SDFG_ENABLE_COVERAGE`          | `OFF`     | Build with coverage instrumentation                                             |
| `SDFG_BUILD_DOCS`               | `OFF`     | Build the Doxygen documentation                                                 |
| `RELEASE_PACKAGE`               | `OFF`     | Use the plain version number instead of appending the Git hash                  |
| `PACKAGE_WITH_TOOL_DEPS`        | `OFF`     | Detect runtime dependencies and add them to the package metadata (used for releases) |

## Debug builds

For development, build with debug information, tests, and a sanitizer:

```bash
cmake -G Ninja -B build-debug \
    -DCMAKE_C_COMPILER=clang-21 \
    -DCMAKE_CXX_COMPILER=clang++-21 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_BUILD_FRONTEND=ON \
    -DSDFG_ENABLE_SANITIZER=ON \
    -DSDFG_SANITIZER=address
ninja -C build-debug
```

[Debug Options](./debugging.md) explains how to run the tests and inspect what docc does with your code.
