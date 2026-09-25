---
title: Commandline Options
sidebar_position: 1
description: How to use docc from the commandline
---

`docc` is based on Clang/LLVM, so most of their options are inherited. Because of the way docc plugs into LLVM, some options may interfere with docc's operation. docc picks relevant parts of the original source code to handle itself, partially outside of LLVM's control, before reintegrating the further optimized results.

For this, new source files may be generated at link time that contain that code. This link-time generated code can have different build options than the original source code. docc generally manages the build options so that the code can interoperate.

## Environment options

- `DOCC_TMP`: where docc puts its intermediate outputs. Defaults to `/tmp/[$USER]/DOCC`. During normal operation, docc removes its intermediate outputs when it's done, like other compilers.

## Optimization options

### `-docc-tune=...`

Chooses the target to optimize for. See [Targets](./targets.md).

- Default: `none`
- Options:
  - `sequential`
  - `openmp`: parallelize across multiple CPU cores using OpenMP
  - `cuda`: offload to NVIDIA CUDA accelerators and GPUs
  - `tenstorrent`: offload to Tenstorrent Wormhole and Blackhole accelerator cards

### `-docc-offloading-force-synchronous`

For offloading backends that support it, use synchronous instead of asynchronous kernel calls. Host profiling then includes processing time on the offload device, at the cost of performance. Without it, host-to-device and device-to-host transfer time measurements may be meaningless because the transfers happen in the background.

### `-docc-no-offloading-transfer-opt`

Disable attempts to remove unnecessary transfers between host and offload device. The optimization can save significant time, but may hide bugs because transfers may no longer happen near the code working on the data.

### `-docc-transfer-tune`

Enable transfer tuning, which queries the Daisytuner cloud database for possible optimizations. Access requires a valid registration/license of your device. The level of access to transfer tuning for different targets may depend on the linked account.

### `-docc-func-blacklist=...`

Sets a regular expression for functions that should not be optimized beyond Clang's normal handling. Helpful when setting experimental options that only work for some parts of the code. Matches the final symbol names (mangled names for C++).

- Example: `'.*blacklisted_function.*'`

### `-docc-lower-invoke`

Removes exception handling code. Such code may be too complex to offload or constrain the code too much for further optimization.

## Handling options

### `-docc-work-dir=...`

Sets a specific output directory for intermediate files instead of generating a unique directory under `DOCC_TMP`. Implies `-docc-save-temps`.

### `-docc-save-temps`

Stops docc from deleting its intermediate output directory.

### `-docc-comp-opt=...`

Sets compile options for newly generated code manually. Can be repeated as often as needed.

- When set at compile time of a code unit, it is inherited by any code generated from that unit.
- When set at link time, it takes precedence over all other manually set options of the different code units.

Most commonly used to build generated code with debug information: `-docc-comp-opt=-g`. The main `-g` flag is not applied to generated code by default, because that debug information would point to machine-generated source files in temporary directories rather than your original sources.

## Plugins

### `-docc-plugins=...`

Comma-separated list of docc plugins. Each plugin can add domain- or application-specific knowledge for further optimization. This may include replacing entire functions with code better suited to offloading, or using different data representations.

See [docc Plugins](./plugins.md).

## Instrumentation

### `-docc-instrument=...`

Enables instrumentation of generated code, which measures performance data at runtime.

- Default: `none`
- Options:
  - `ols`: instruments only the outermost code in each generated code region.

See [Runtime Instrumentation](./runtime-instrumentation.md) for how to run instrumented code and control its measurements and outputs.
