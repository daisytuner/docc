---
title: Debug Options
sidebar_position: 4
description: Inspect what docc does with your code, debug generated code, and run the test suites
---

This page is for developers who want to understand why docc optimizes code the way it does, or who work on docc itself. We continue with WAXPBY from the [Quickstart](./quickstart/first-program.md) and ask three questions: did docc pick up the kernel, what did it do with it, and is the generated code correct?

## Keep the intermediate files

docc writes intermediate files to a temporary directory under `DOCC_TMP` and deletes them after compilation. Keep them in a directory of your choice:

```bash
docc -O3 -docc-tune=cuda -docc-work-dir=./docc-out waxpby.c -o waxpby
```

| Flag                     | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| `-docc-work-dir=<dir>`   | Write intermediate files to `<dir>`. Implies `-docc-save-temps`.            |
| `-docc-save-temps`       | Keep the intermediate files in `DOCC_TMP` instead of deleting them.         |

The work directory contains one subdirectory per translation unit with the SDFGs (JSON), the generated sources, and the reports below.

## Did docc pick up the kernel?

docc only optimizes code that it can *lift* into its intermediate representation. The lifting report lists every lifted region and, for every failure, the source location and the reason:

```bash
DOCC_LIFTING_REPORT=1 docc -g -O3 -docc-tune=cuda -docc-work-dir=./docc-out waxpby.c -o waxpby
cat docc-out/*/lifting_report.json
```

Compile with `-g` so that the report contains source locations. Each failure names the LLVM instruction that docc does not support, for example a call to an external function, an unsupported intrinsic, or exception handling (try [`-docc-lower-invoke`](./usage.md#control-the-optimization)).

## What did docc do with it?

| Option                         | Output                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| `DOCC_OFFLOAD_REPORT=1`        | `*.opt_report.json`: per loop nest, which targets were tested and which was chosen      |
| `-docc-dump-sdfg`              | The LLVM IR of each lifted function (`.ll`) and the SDFG after each step (`.dot`) |
| `-docc-dot-scheduled`          | The SDFG after scheduling for the target (`.dot`)                                      |
| `-docc-dump-glbl-cfg[=<dir>]`  | The global control-flow graph across all modules at link time                          |
| `DOCC_STATISTICS=1`            | Runtime and metrics of the individual compiler passes                                  |

Render the DOT files with Graphviz:

```bash
docc -O3 -docc-tune=cuda -docc-work-dir=./docc-out -docc-dump-sdfg -docc-dot-scheduled waxpby.c -o waxpby
dot -Tsvg docc-out/<module>/<sdfg>.dot -o sdfg.svg
```

## Is the generated code correct?

| Option                                 | Use it to                                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `-docc-comp-opt=-g`                    | Compile the generated code with debug information. The main `-g` does not apply to generated code.  |
| `-docc-offloading-force-synchronous`   | Make all kernel launches and transfers synchronous, so errors and timings appear where they happen. |
| `-docc-no-offloading-transfer-opt`     | Keep all host-device transfers next to the code that uses the data.                                 |
| `-docc-func-blacklist=<regex>`         | Exclude functions from optimization to narrow down which one causes a problem.                     |
| `-docc-capture-args`                   | Save the inputs and outputs of each optimized region to compare them with a reference run.         |
| `-docc-noop`                           | Skip docc entirely and only run Clang, to check whether a problem is caused by docc at all.         |

### GPU error checking

By default, the generated GPU code does not check the return values of CUDA and HIP calls, so errors can surface much later or not at all. Set `DOCC_CUDA_DEBUG=1` (NVIDIA) or `DOCC_ROCM_DEBUG=1` (AMD) **at compile time** to generate code that checks every API call, synchronizes after every kernel launch, and exits with the error and its location:

```bash
DOCC_CUDA_DEBUG=1 docc -O3 -docc-tune=cuda waxpby.c -o waxpby
./waxpby
# CUDA error: an illegal memory access was encountered File: ..., Line: ...
```

The checks are compiled into the program, so recompile without the variable for production builds.

### Step through generated code

For example, to step through the generated WAXPBY kernel in a debugger:

```bash
docc -g -O0 -docc-tune=openmp -docc-work-dir=./docc-out -docc-comp-opt=-g waxpby.c -o waxpby
gdb ./waxpby
```

## Debug builds and tests

Build docc in debug mode, optionally with a sanitizer, as described in [Custom Installation](./custom-installation.md#debug-builds). In debug builds, LLVM's debug output of docc's passes is available with `-mllvm -debug-only=docc`.

Run the test suites from the build directory:

```bash
# Unit tests of all modules
ctest --test-dir build --parallel

# Tests of the LLVM plugin
./build/llvm/tests/docc_llvm_pass_test

# End-to-end tests (requires an installed docc package)
cd llvm/integration && pytest -v llvm_test_suite.py
```

To build the unit tests, configure with `-DSDFG_BUILD_TESTS=ON -DLLVM_BUILD_TESTS=ON` (the default).
