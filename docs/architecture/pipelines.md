---
title: Compilation Pipelines
sidebar_label: Pipelines
sidebar_position: 2
description: How C/C++, Python, and PyTorch programs flow through docc
---

This page follows a program through docc, for each frontend. All three pipelines meet in the same place: a Structured SDFG that is optimized, scheduled for a target, and turned into C/C++ code by the core. The components are described in the [Architecture Overview](./overview.md).

## C/C++

The C/C++ frontend hooks into Clang and LLVM. `docc` and `docc-cpp` are thin drivers around `clang-21`/`clang++-21` (`llvm/driver/driver.cpp`). They load the docc LLVM plugin (`-fpass-plugin`), enable ThinLTO (`-flto=thin`), and use `docc-ld` as linker. Work is split between compile time and link time:

```text
 Compile time (per source file)                Link time (whole program)
 ──────────────────────────────                ─────────────────────────
 source ─► Clang ─► LLVM IR                    docc-ld ─► ThinLTO
                      │                                     │
                      ▼                                     ▼
 lifting          LLVM functions ─► SDFGs      argument expansion: decide data
                      │                        placement across modules and calls
                      ▼                                     │
 optimization     normalization, einsum                     ▼
                      │                        code generation: SDFG ─► C/C++/CUDA
                      ▼                                     │
 scheduling       loops ─► target schedules                 ▼
                      │                        c-compile: compile generated code,
                      ▼                        link it into the executable
                  SDFGs + reports in work dir
```

| Stage              | Where                                                         | What happens                                                                                     |
| ------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Lifting            | `llvm/src/lifting/`, `FunctionToSDFGPass`                      | Translates LLVM functions and loops into SDFGs. Unsupported code stays LLVM IR.                  |
| Optimization       | `NormalizationPass`, `EinsumPass`                              | Normalizes loops, optionally detects tensor contractions                                         |
| Scheduling         | `SchedulingPass`, target plugins in `opt/`, `targets/`         | Selects a schedule per loop nest for the target set with `-docc-tune`                            |
| Argument expansion | `ArgumentExpansionPass`                                        | At link time, sees all modules and decides where data lives, which removes redundant transfers  |
| Code generation    | `CodeGenerationPass`, `sdfg/src/codegen/`, `c-compile/`        | Generates source code for each SDFG, compiles it for the target, and links it                     |

The SDFGs are stored in the work directory between compile and link time. The link step finds them through the ThinLTO index (`SDFGRegistry` in `llvm/src/analysis/`). This is how docc optimizes across files and libraries, as in [Using Libraries](../c-cpp/quickstart/libraries.md).

## Python

The Python frontend runs when a `@native` function is called for the first time (`python/docc/python/python_program.py`):

```text
 Python function + arguments
          │
          ▼
 ASTParser          Python AST ─► Structured SDFG                     py0.parsed
          │
          ▼
 expansion          NumPy operations (library nodes) ─► loops         py2.expanded
          │
          ▼
 optimization       simplification, loop fusion                       py3.opt
          │
          ▼
 normalization      loop normalization (CPU targets)                  py4.norm
          │
          ▼
 scheduling         loops ─► target schedules                         py5.post_sched
          │
          ▼
 code generation    SDFG ─► C++/CUDA/HIP ─► Clang ─► shared library
          │
          ▼
 CompiledSDFG       loads the library and calls it with NumPy/CuPy arrays
```

The names on the right are the stages written with `DOCC_DEBUG=dump`, see [Python Debug Options](../python/debugging.md). The pipeline itself (`DoccProgram.sdfg_pipe` in `python/docc/compiler/docc_program.py`) is shared with the PyTorch backend. The heavy lifting happens in C++, through the bindings in `python/bindings/`.

## PyTorch

The PyTorch backend replaces the Python parser with a graph parser and then reuses the Python pipeline:

```text
 nn.Module
     │
     ▼
 TorchDynamo            torch.compile captures an FX graph
     │
     ▼
 torch.export           FX graph ─► Core ATen IR
     │
     ▼
 GraphParser            one handler per ATen operator ─► Structured SDFG
     │
     ▼
 Python pipeline        expansion, optimization, scheduling, code generation (see above)
     │
     ▼
 CompiledSDFG           called with torch tensors (CPU or GPU)
```

The entry point is `PyTorchProgram` in `pytorch/docc/pytorch/pytorch_program.py`, registered as the `docc` backend on import. The operator handlers live in `pytorch/docc/pytorch/graph_parser/`; to support a new operator, add a handler there and a test in `pytorch/tests/ops/`.

Earlier versions translated PyTorch models via torch-mlir and the MLIR frontend in `mlir/`. This path is no longer used.
