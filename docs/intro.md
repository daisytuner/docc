---
title: Introduction
sidebar_position: 1
slug: /intro
description: What docc is, what it offers, and where to start
---

The Daisytuner Optimizing Compiler Collection (**docc**) compiles ordinary C/C++, Python/NumPy, and PyTorch code for multi-core CPUs, GPUs, and accelerators. You keep your code as it is and choose a target; docc parallelizes, offloads, and optimizes it for that target.

## What docc offers

- **Drop-in C/C++ compiler**: use `docc` instead of `clang` or `gcc`.
- **Python JIT**: compile NumPy functions with the `@native` decorator.
- **PyTorch backend**: run models through `torch.compile(..., backend="docc")`.
- **Automatic parallelization and offloading**: OpenMP, CUDA, ROCm, and Tenstorrent from the same source code.
- **Data-transfer optimization**: avoids unnecessary copies between host and device, even across library boundaries.
- **Runtime instrumentation**: measure performance counters and capture kernel arguments of generated code.
- **Transfer tuning** (optional): reuse optimizations from the Daisytuner cloud database.

## Same idea, three frontends

Each example parallelizes the same computation across CPU cores with OpenMP. Change the target to `cuda` or `rocm` to run it on a GPU.

**C/C++**

```bash
docc -O3 -docc-tune=openmp app.c -o app
```

**Python**

```python
from docc.python import native

@native(target="openmp")
def matmul(A, B):
    return A @ B
```

**PyTorch**

```python
import docc.pytorch

compiled = torch.compile(model, backend="docc", options={"target": "openmp"})
```

## How it works

All frontends translate code into one shared intermediate representation, the **stateful dataflow multigraph (SDFG)**. docc analyzes and optimizes the SDFG, then generates code for the selected hardware:

```text
Frontends          Intermediate representation       Targets

┌───────────┐
│ C/C++     │──┐                                     ┌──────────────────────────────────┐
└───────────┘  │   ┌─────────────────────────┐   ┌──►│ CPU          sequential, OpenMP  │
┌───────────┐  │   │ SDFG IR                 │   ├──►│ NVIDIA GPU   CUDA                │
│ Python    │──┼──►│ analysis & optimization │───┼──►│ AMD GPU      ROCm                │
└───────────┘  │   │ code generation         │   ├──►│ Tenstorrent  Wormhole, Blackhole │
┌───────────┐  │   └─────────────────────────┘   └──►│ Apple GPU    Metal (in progress) │
│ PyTorch   │──┘                                     └──────────────────────────────────┘
└───────────┘
```

[Architecture & Development](./architecture/overview.md) describes the components in detail.

## Compatibility

|                 | OpenMP | CUDA | ROCm | Tenstorrent | Metal |
| --------------- | :----: | :--: | :--: | :---------: | :---: |
| C/C++ (Linux)   |   ✅   |  ✅  |  ✅  |     ✅      |   —   |
| Python (Linux)  |   ✅   |  ✅  |  ✅  |      —      |   —   |
| PyTorch (Linux) |   ✅   |  ✅  |  ✅  |      —      |   —   |
| Python (macOS)  |   ✅   |  —   |  —   |      —      |  🚧   |
| PyTorch (macOS) |   🚧   |  —   |  —   |      —      |  🚧   |

✅ Supported | 🚧 Work in progress

## Where to go next

- **C/C++**: [Quickstart](./c-cpp/quickstart/install.md)
- **Python**: [Quickstart](./python/quickstart/install.md)
- **PyTorch**: [Quickstart](./pytorch/quickstart/install.md)
- **Contributors**: [Architecture & Development](./architecture/overview.md)
- Daisy Cloud, self-hosted runners, and performance guides are covered in the [Daisytuner Docs](https://docs.daisytuner.com).
