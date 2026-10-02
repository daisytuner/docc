---
title: Introduction
sidebar_position: 1
slug: /intro
description: What docc is and how it is organized
---

The Daisytuner Optimizing Compiler Collection (**docc**) implements an intermediate representation as well as frontends, drivers, and code generation for translating and optimizing various programming languages for multiple targets.

The core of the project is **stateful dataflow multigraphs (SDFGs)**, implemented in the `sdfg` module. It contains the definition of the intermediate representation as well as numerous passes and analyses. For instance, docc supports auto-parallelization using data-centric and polyhedral analysis.

SDFGs can be generated from Python (JIT) and MLIR frontends, which are separate components including Python bindings and an MLIR dialect for conversion. Targets such as Generic, [OpenMP](https://www.openmp.org/), [CUDA](https://developer.nvidia.com/cuda/toolkit), and [ROCm](https://rocmdocs.amd.com/en/latest/) are implemented in `opt`.

The repository also contains runtime libraries for code instrumentation (performance counters and data capturing).

## Compatibility

|                 | OpenMP | CUDA | ROCm | Metal |
| --------------- | :----: | :--: | :--: | :---: |
| C/C++ (Linux)   |   ✅   |  ✅  |  ✅  |   —   |
| Python (Linux)  |   ✅   |  ✅  |  ✅  |   —   |
| PyTorch (Linux) |   ✅   |  ✅  |  ✅  |   —   |
| Python (macOS)  |   ✅   |  —   |  —   |  🚧   |
| PyTorch (macOS) |   🚧   |  —   |  —   |  🚧   |

✅ Supported | 🚧 Work in progress

## Repository layout

| Module           | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `sdfg`           | SDFG intermediate representation, passes, analyses, code generation |
| `opt`            | Transformations and target backends (OpenMP, CUDA, ROCm, …)         |
| `llvm`           | LLVM-based C/C++ frontend (lifting LLVM IR to SDFGs)                |
| `mlir`           | SDFG MLIR dialect and conversion from core MLIR dialects            |
| `python`         | Python bindings and the `@native` JIT frontend                      |
| `pytorch`        | PyTorch `torch.compile` backend                                     |
| `c-compile`      | Compiler driver                                                     |
| `rtl`            | Runtime library for instrumentation                                 |
| `arg-capture-io` | Runtime data capturing                                              |
| `rpc`            | Remote optimization passes                                          |
| `targets`        | Additional offload targets                                          |

The C++ headers of all modules are documented in the [C++ API reference](./reference/api.md).

## Where to go next

- C/C++: [Install docc](./getting-started-docc/install.md) and [run your first code](./getting-started-docc/first-program.md).
- Python: [Install docc for Python](./getting-started-python/install.md).
- PyTorch: [Use docc as a `torch.compile` backend](./getting-started-pytorch/usage.md).
- Daisy Cloud, self-hosted runners, and performance guides are covered in the [Daisytuner Docs](https://docs.daisytuner.com).
