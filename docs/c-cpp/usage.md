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

| Flag                                      | Description                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-docc-func-blacklist=<regex>`            | Do not optimize functions whose full symbol name (mangled for C++) matches the regex, for example `'init'` or `'.*_ref$'`.                                                                                                                                                                                                                |
| `-docc-lower-invoke`                      | Remove exception-handling code before optimizing. Such code can otherwise prevent offloading.                                                                                                                                                                                                                                             |
| `-docc-offload-unknown-sizes`             | Offload arrays whose size docc cannot determine, using `malloc_usable_size()` as an upper bound.                                                                                                                                                                                                                                          |
| `-docc-no-offloading-transfer-opt`        | Do not remove redundant host-device transfers. Slower, but transfers stay close to the code that uses the data.                                                                                                                                                                                                                           |
| `-docc-einsum`                            | Detect loops that implement tensor contractions (Einstein notation) and optimize them as such.                                                                                                                                                                                                                                            |
| `-docc-plugins=<list>`                    | Load a comma-separated list of [plugins](../architecture/plugins.md) with domain-specific optimizations.                                                                                                                                                                                                                                  |
| `-docc-offloading-force-synchronous`      | For supporting offloading backends, use synchronous kernel calls instead of asynchronous kernel calls. Makes host profiling include any processing time on the offloaded device at the cost of performance. Without it, host-to-device and device-to-host transfer time measurements may be meaningless as they happen in the background. |
| `-docc-expand=none\|all` (default `none`) | Overrides when to expand library nodes.                                                                                                                                                                                                                                                                                                   |
| `-docc-no-inline-functions`               | Disables inlining of functions during DOCC extraction.                                                                                                                                                                                                                                                                                    |

For example, to offload WAXPBY but keep the initialization loop on the CPU, move the initialization into a function `init` and run:

```bash
docc -O3 -docc-tune=cuda -docc-func-blacklist='init' waxpby.c -o waxpby
```

## General helpers

| Flag                              | Description                                                                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-docc-root=<path>`               | Root of DOCC. Can be `CMake:[path-to-cmake-build-dir]` or `Dist:[path-to-docc-dist-dir]`.                                                                              |
| `-docc-work-dir=<path>`           | Sets a specific output directory for the intermediary files, instead of generating a unique directory underneath the `DOCC_TMP` directory. Implies `-docc-save-temps`. |
| `-docc-save-temps`                | Stops docc from deleting its intermediate output directory.                                                                                                            |
| `-docc-dot-scheduled`             | Dumps the scheduled graph to a DOT file.                                                                                                                               |
| `-docc-dump-sdfg`                 | Enables output of sdfgs.                                                                                                                                               |
| `-docc-link`                      | **Experimental**: Sets special options for how to link.                                                                                                                |
| `-docc-combined-index=<path>`     | Thin-link `.index.bc` output file.                                                                                                                                     |
| `-docc-combined-lib-index=<path>` | Attributes for external function from shared objects.                                                                                                                  |
| `-docc-dot-codegen`               | Write out every SDFG for which we generate code to a DOT file for debugging purposes.                                                                                  |

## Global CFG

| Flag                                      | Description                                                                                                                       |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `-docc-dump-glbl-cfg=<path>`              | Path to dump global cfgs.                                                                                                         |
| `-docc-debug-glbl`                        | Enables debug messages for global CFG analysis.                                                                                   |
| `-docc-include-all-intrinsics`            | Add external call nodes for all intrinsics in global CFG analysis, e.g. `llvm.dbg.value`                                          |
| `-view-globalcfg-nodes=.*`                | For each entry-point dump reachable global CFG nodes that match the regex.                                                        |
| `-print-globalcfgs=.*`                    | Print dot files of global CFG from the entry-points that match the regex.                                                         |
| `-docc-globalcfg-lib-entry-points=<path>` | Path to a file containing a list of function names, defining relevant entry points into the globalcfg. `main` is always included. |

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

If the PAPI library cannot be loaded, `__DAISY_PAPI_VERSION` is unset or does not match it, or a listed event does not exist, the instrumented program stops at its first measured region with `[daisy-rtl] Instrumentation runtime not ready: <reason>` (SIGABRT, exit code 134).

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
