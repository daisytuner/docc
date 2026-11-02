---
title: Usage
sidebar_position: 3
description: The @native decorator, its options, GPU data residency, caching, and environment variables
---

This page explains how to control docc from Python, using the WAXPBY example from the [Quickstart](./quickstart/first-program.md).

## The `@native` decorator

Use `@native` with or without options. Options are keyword arguments:

```python
from docc.python import native

@native
def f(x): ...

@native(target="openmp", category="desktop")
def waxpby(alpha, x, beta, y, w):
    for i in range(x.shape[0]):
        w[i] = alpha * x[i] + beta * y[i]
```

A decorated function compiles on its first call. To compile ahead of time, for example outside a timed region, call `compile` with example arguments. It returns the compiled function:

```python
kernel = waxpby.compile(2.0, x, 3.0, y, w)   # compiles, does not run
kernel(2.0, x, 3.0, y, w)                    # runs
```

## Choose a target

| Target       | Hardware           | Description                                                     |
| ------------ | ------------------ | --------------------------------------------------------------- |
| `none`       | single CPU core    | Default. Compiles the function as-is, without parallelization. |
| `sequential` | single CPU core    | Optimizes and vectorizes the code.                              |
| `openmp`     | all CPU cores      | Parallelizes loops with OpenMP.                                 |
| `cuda`       | NVIDIA GPUs        | Offloads loops to the GPU. Requires CuPy.                       |
| `rocm`       | AMD GPUs           | Offloads loops to the GPU. Requires CuPy.                       |

`category` tells the scheduler which class of hardware to optimize for: `edge`, `desktop`, or `server` (default).

## Keep data on the GPU

When you call a GPU function with NumPy arrays, docc copies the arrays to the GPU and back on every call, and emits a `DoccPerformanceWarning`. To avoid these copies, compile the function with NumPy arrays and then call it with [CuPy](https://cupy.dev/) arrays. The data stays on the GPU between calls:

```python
import cupy as cp

kernel = waxpby.compile(2.0, x, 3.0, y, w)   # compile with NumPy arrays

x_d, y_d = cp.asarray(x), cp.asarray(y)      # copy inputs to the GPU once
w_d = cp.zeros_like(x_d)
v_d = cp.zeros_like(x_d)

kernel(2.0, x_d, 3.0, y_d, w_d)              # w = 2x + 3y, on the GPU
kernel(2.0, w_d, 3.0, y_d, v_d)              # v = 2w + 3y, w never leaves the GPU
v = cp.asnumpy(v_d)
```

All arrays of a call must be either NumPy or CuPy arrays. If a function cannot run entirely on the GPU, for example because a loop has a dependency between iterations, docc runs that part on the host and warns that device residency could not be enabled.

## Options

| Option                  | Default             | Description                                                                                                       |
| ----------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `target`                | `"none"`            | Target, see above                                                                                                 |
| `category`              | `"server"`          | Hardware class: `edge`, `desktop`, `server`                                                                       |
| `remote_tuning`         | `False`             | Query the Daisytuner optimization database for known-good optimizations (transfer tuning)                         |
| `symbolic_shapes`       | `True`              | Compile for arbitrary array sizes. With `False`, the sizes of the first call are compiled in.                     |
| `approximate_math`      | `False`             | Use fast, approximate math functions on GPUs (for example `__expf`)                                               |
| `fp_contract`           | `True`              | Fuse multiplications and additions into fused multiply-adds, like `-ffp-contract=fast`                            |
| `contraction_precision` | target default      | Precision of float matrix multiplications: `native`, `int2` … `int8` (quantized), `ozaki`, `ozaki1[:s]`, `ozaki2[:L]` (emulated on Int8 units) |
| `einsum`                | —                   | Detect loops that implement tensor contractions and optimize them as such                                         |
| `normalize`             | CPU targets: `True` | Normalize loops before scheduling                                                                                 |
| `device_residency`      | GPU targets: `True` | Allow device-resident arguments, see above                                                                         |
| `instrumentation_mode`  | `""`                | `"ols"` or `"ols_tuned"` to measure each optimized loop nest, see [C/C++ instrumentation](../c-cpp/usage.md#measure-performance) |
| `capture_args`          | `False`             | Save the inputs and outputs of each optimized region                                                              |
| `pass_options`          | `{}`                | Options of individual compiler passes, see below                                                                  |

Options for debugging are described in [Debug Options](./debugging.md). Unknown options are ignored with a warning.

### Pass options

Individual compiler passes register their own options. List them with:

```python
from docc.sdfg import registered_options

for opt in registered_options():
    print(opt["key"], opt["type"], opt["default"], opt["doc"])
```

Pass them by key, either in `pass_options` or directly as keyword arguments:

```python
@native(target="openmp", pass_options={"symbolic.bound_budget": 100000})
def waxpby(alpha, x, beta, y, w): ...
```

## Caching

docc caches compiled functions in memory, per combination of options and argument types. With `symbolic_shapes=True`, arrays of different sizes reuse the same compiled code.

The compiled libraries are stored under `$DOCC_TMP` (default `/tmp/$USER/DOCC`). To reuse them across Python processes, for example in a benchmark script that you run repeatedly, set `DOCC_REUSE_BINARIES=1`.

## Supported Python

docc compiles a subset of Python that is common in numerical code:

- Scalars (`int`, `float`, `bool`, NumPy scalar types) and NumPy arrays as arguments
- `for` loops over `range`, `if`/`else`, `while`
- Array indexing and slicing, `x.shape`
- Arithmetic, `math` functions, and many NumPy functions and operators, for example `@`, `np.sum`, `np.exp`, `np.einsum`
- Calls to other Python functions, which are inlined

The function source must be available to `inspect`, so functions defined in an interactive shell or with `exec` cannot be compiled.

## Environment variables

| Variable                | Description                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `DOCC_TMP`              | Directory for generated code and compiled libraries, default `/tmp/$USER/DOCC`            |
| `DOCC_BACKEND_COMPILER` | C++ compiler for generated code, default `clang++-21` (Linux) or `clang++` (macOS)        |
| `DOCC_CUDA_ARCH`        | NVIDIA GPU architecture, for example `sm_90`. Default: the first GPU in the system       |
| `DOCC_ROCM_ARCH`        | AMD GPU architecture, for example `gfx942`. Default: the first GPU in the system         |
| `ROCM_PATH`             | ROCm installation, default `/opt/rocm`                                                    |
| `DOCC_REUSE_BINARIES`   | `1` to reuse compiled libraries from earlier runs                                         |
| `DOCC_REUSE_SOURCES`    | `1` to reuse generated sources from earlier runs and only recompile them                  |

Explicit decorator options take precedence over environment variables.

## Next step

If docc does not compile or optimize your function as expected, [Debug Options](./debugging.md) shows how to look at what docc does with WAXPBY.
