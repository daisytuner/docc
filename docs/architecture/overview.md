---
title: Architecture Overview
sidebar_label: Overview
sidebar_position: 1
description: How the docc repository is structured and how its components work together
---

docc consists of three frontends, one shared core, and a set of target backends. All frontends translate code into the same intermediate representation, the **stateful dataflow multigraph (SDFG)**. Everything after that, from analysis to code generation, is shared. This page describes the components and how they depend on each other. [Compilation Pipelines](./pipelines.md) follows a program through them.

## Components

```text
            ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
Frontends   │ llvm/              │ │ python/            │ │ pytorch/           │
            │ C/C++ (docc)       │ │ @native, bindings  │ │ torch.compile      │
            └────────────────────┘ └────────────────────┘ └────────────────────┘
                                              ▼
            ┌──────────────────────────────────────────────────────────────────┐
Core        │ sdfg/  SDFG IR, builders, analyses, passes, codegen              │
            └──────────────────────────────────────────────────────────────────┘
                                              ▲ plugins
            ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
Targets     │ opt/               │ │ targets/           │ │ rpc/               │
            │ OpenMP, CUDA, ROCm │ │ Tenstorrent, ETSoC │ │ remote tuning      │
            └────────────────────┘ └────────────────────┘ └────────────────────┘
                                              ▼
            ┌──────────────────────────────────────────────────────────────────┐
Build       │ c-compile/  target registry, compiles generated code             │
            └──────────────────────────────────────────────────────────────────┘
                                              ▼
            ┌────────────────────┐ ┌────────────────────┐
Runtime     │ rtl/               │ │ arg-capture-io/    │
            │ instrumentation    │ │ argument capture   │
            └────────────────────┘ └────────────────────┘
```

- **Frontends** translate source code into SDFGs. `pytorch/` is written in Python and builds on `python/`.
- **Core** (`sdfg/`) defines the SDFG, provides the builders that frontends use, and contains the analyses, generic passes, and the code generator.
- **Targets** add target-specific transformations and code generation as plugins of the core: `opt/` for the built-in targets, `targets/` for accelerator backends, and `rpc/` for optimizations that are queried from the Daisytuner service.
- **Build** (`c-compile/`) knows how to compile generated code for each target (compiler flags, libraries, GPU architectures) and registers the built-in targets.
- **Runtime** libraries are linked into programs that use instrumentation or argument capture.

## Repository layout

| Directory              | Contents                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| `sdfg/`                | SDFG intermediate representation, builders, analyses, passes, serialization, code generation              |
| `opt/`                 | Transformations and built-in targets: parallelization, tiling, vectorization, einsum, OpenMP, CUDA, ROCm |
| `rpc/`                 | Remote optimization passes (transfer tuning) that query the Daisytuner service                            |
| `targets/`             | Accelerator backends as plugins: Tenstorrent (`tenstorrent/`), ETSoC (`et/`)                              |
| `c-compile/`           | Target registry; compiles and links the generated code                                                    |
| `llvm/`                | C/C++ frontend: compiler drivers (`docc`, `docc-cpp`, `docc-ld`) and the LLVM plugin                      |
| `python/`              | Python bindings of the core and the `@native` frontend (package `docc-compiler`)                          |
| `pytorch/`             | `torch.compile` backend (package `docc-pytorch`)                                                          |
| `rtl/`                 | Runtime library for instrumentation (timers, PAPI counters, traces)                                       |
| `arg-capture-io/`      | Reading and writing captured kernel arguments                                                             |
| `mlir/`                | Legacy MLIR frontend, formerly used for PyTorch; only built with `-DMLIR_BUILD_FRONTEND=ON`                |
| `tutorial/`            | Example target plugin, see [Adding Custom Offload Targets](./custom-offload-targets.md)                   |
| `examples/`            | Example programs                                                                                          |
| `cmake/`, `scripts/`   | Build helpers and developer scripts                                                                       |

## Inside the core

| Directory in `sdfg/include/sdfg/` | Contents                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `data_flow/`                      | Data containers, memlets (data movement), tasklets (computations), library nodes                       |
| `structured_control_flow/`        | Structured SDFGs: sequences, loops, maps, and branches, see [Structured SDFGs](./structured-sdfgs.md) |
| `builder/`                        | APIs to construct SDFGs, used by all frontends                                                         |
| `analysis/`                       | Data dependencies, loops, memory layout, types                                                         |
| `passes/`                         | Simplifications, data-flow optimizations, scheduling                                                   |
| `codegen/`                        | Generation of C/C++ code, including instrumentation                                                    |
| `serializer/`                     | Reading and writing SDFGs as JSON                                                                      |
| `plugins/`                        | Interface for target plugins                                                                           |

## How targets plug in

A target is a plugin that registers schedules, code generators, and library-node implementations with the core. The built-in targets in `opt/` and the accelerator backends in `targets/` use the same interface (`sdfg/plugins/plugins.h`). External plugins are shared libraries that export `register_docc_plugin` and are loaded with [`-docc-plugins`](../c-cpp/usage.md#control-the-optimization).

For a complete example, see [Adding Custom Offload Targets](./custom-offload-targets.md). The C++ API of all components is documented in the [C++ API reference](../reference/api.md).
