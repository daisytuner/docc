---
title: Usage
sidebar_position: 3
description: Compiler flags, targets, instrumentation, and environment variables of docc for C/C++
---

This page explains how to control docc in your builds, using the WAXPBY example from the [Quickstart](./quickstart/first-program.md). All docc-specific flags start with `-docc-`; all other flags are passed to Clang unchanged.

## Use docc in your build system

`docc` replaces `clang`, and `docc-cpp` replaces `clang++`. Pass docc flags like any other compiler flag:

```bash
# CMake
CC=docc CXX=docc-cpp cmake -B build -DCMAKE_C_FLAGS="-docc-tune=openmp" -DCMAKE_CXX_FLAGS="-docc-tune=openmp"

# Make
make CC=docc CXX=docc-cpp CFLAGS="-O3 -docc-tune=openmp"
```

Use the same `-docc-tune` value at compile and link time. docc generates parts of the optimized code at link time.

## Choose a target

`-docc-tune=<target>` selects the hardware docc optimizes for:

| Target        | Hardware                               | Backends used                  |
| ------------- | -------------------------------------- | ------------------------------ |
| `none`        | default, regular Clang compilation     | —                              |
| `sequential`  | single CPU core                        | sequential C                   |
| `openmp`      | all CPU cores                          | OpenMP                         |
| `cuda`        | NVIDIA GPUs                            | CUDA, OpenMP for host code     |
| `rocm`        | AMD GPUs                               | ROCm/HIP, OpenMP for host code |
| `tenstorrent` | Tenstorrent Wormhole and Blackhole     | TT-Metalium                    |

## Control the optimization

| Flag                                 | Description                                                                                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-docc-func-blacklist=<regex>`       | Do not optimize functions whose full symbol name (mangled for C++) matches the regex, for example `'init'` or `'.*_ref$'`.                                    |
| `-docc-lower-invoke`                 | Remove exception-handling code before optimizing. Such code can otherwise prevent offloading.                                                                      |
| `-docc-offload-unknown-sizes`        | Offload arrays whose size docc cannot determine, using `malloc_usable_size()` as an upper bound.                                                                   |
| `-docc-no-offloading-transfer-opt`   | Do not remove redundant host-device transfers. Slower, but transfers stay close to the code that uses the data.                                                    |
| `-docc-einsum`                       | Detect loops that implement tensor contractions (Einstein notation) and optimize them as such.                                                                     |
| `-docc-plugins=<list>`               | Load a comma-separated list of [plugins](../architecture/plugins.md) with domain-specific optimizations.                                                           |

For example, to offload WAXPBY but keep the initialization loop on the CPU, move the initialization into a function `init` and run:

```bash
docc -O3 -docc-tune=cuda -docc-func-blacklist='init' waxpby.c -o waxpby
```

## Transfer tuning

With `-docc-transfer-tune`, docc queries the Daisytuner optimization database for known-good optimizations of similar code. This requires a registered device. `-docc-target-architecture=edge|desktop|server` (default `server`) selects the class of hardware to tune for.

```bash
docc -O3 -docc-tune=openmp -docc-transfer-tune waxpby.c -o waxpby
```

## Options for generated code

docc compiles the code it generates separately from your sources. `-docc-comp-opt=<flag>` adds a compiler flag to this generated code and can be repeated:

```bash
docc -O3 -docc-tune=openmp -docc-comp-opt=-march=native waxpby.c -o waxpby
```

When set at link time, `-docc-comp-opt` overrides the values given at compile time.

## Measure performance

docc can instrument the code it generates and record runtimes and hardware counters for each optimized region.

| Flag                         | Description                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| `-docc-instrument=ols`       | Measure every outermost loop that docc optimized, including host-device transfers.                     |
| `-docc-instrument=ols_tuned` | Like `ols`, but groups loops that originate from the same source location.                              |
| `-docc-capture-args`         | Save the inputs and outputs of each optimized region to disk, for example to replay or verify a kernel. |

Instrumented programs use [PAPI](https://icl.utk.edu/papi/) at runtime. Build and run WAXPBY with instrumentation:

```bash
docc -O3 -docc-tune=openmp -docc-instrument=ols waxpby.c -o waxpby

export __DAISY_PAPI_VERSION=0x07010000          # version of your PAPI library, here 7.1
export __DAISY_INSTRUMENTATION_EVENTS=PAPI_TOT_CYC,PAPI_L2_TCM
./waxpby                                         # writes daisy_trace.json
```

The trace is a JSON file in Chrome trace format and can be viewed in [Perfetto](https://ui.perfetto.dev/) or `chrome://tracing`.

Runtime settings for instrumented programs:

| Variable                                | Description                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------- |
| `__DAISY_PAPI_VERSION`                  | PAPI version number (required), for example `0x07010000`                            |
| `DAISY_PAPI_PATH`                       | Path to the PAPI library, default `libpapi.so`                                      |
| `__DAISY_INSTRUMENTATION_FILE`          | Output file, default `daisy_trace.json`                                             |
| `__DAISY_INSTRUMENTATION_MODE`          | `aggregate` to record one summary per region instead of one event per execution     |
| `__DAISY_INSTRUMENTATION_EVENTS`        | Comma-separated PAPI events to record on the CPU                                    |
| `__DAISY_INSTRUMENTATION_EVENTS_CUDA`   | Comma-separated PAPI events to record on NVIDIA GPUs                                |
| `__DAISY_CAPTURE_BASE_DIR`              | Output directory of `-docc-capture-args`, default `arg_captures`                    |
| `__DAISY_CAPTURE_STRATEGY_DEFAULT`      | Which executions to capture: `once` (default), `all`, or `never`                    |

## Environment variables

| Variable          | Description                                                                         |
| ----------------- | ----------------------------------------------------------------------------------- |
| `DOCC_TMP`        | Directory for intermediate files, default `/tmp/$USER/DOCC`                         |
| `DOCC_CUDA_ARCH`  | NVIDIA GPU architecture, for example `sm_90`. Default: the first GPU in the system |
| `DOCC_ROCM_ARCH`  | AMD GPU architecture, for example `gfx942`. Default: the first GPU in the system   |
| `ROCM_PATH`       | ROCm installation, default `/opt/rocm`                                              |
| `TT_METAL_INSTALL_DIR`, `TT_METAL_HOME` | TT-Metalium installation for the `tenstorrent` target (one is required) |

## Next step

If docc does not optimize your code as expected, [Debug Options](./debugging.md) shows how to look at what docc does with WAXPBY.
