---
title: Run Your First Code
sidebar_label: First Program
sidebar_position: 2
description: Compile a NumPy function with @native and run it on CPU cores and GPUs
---

Throughout this guide, we use one running example: **WAXPBY**, the vector update `w = alpha * x + beta * y`. It is the same kernel as in the [C/C++ guide](../../c-cpp/quickstart/first-program.md), written as a plain Python function on NumPy arrays.

```python title="waxpby.py"
import numpy as np
from docc.python import native

@native(target="openmp")
def waxpby(alpha, x, beta, y, w):
    for i in range(x.shape[0]):
        w[i] = alpha * x[i] + beta * y[i]

N = 1 << 24
x = np.arange(N, dtype=np.float32)
y = np.arange(N, 0, -1, dtype=np.float32)
w = np.zeros(N, dtype=np.float32)

waxpby(2.0, x, 3.0, y, w)
print(w[0], w[-1])
```

## Step 1: Run it

```bash
python waxpby.py
```

The first call of `waxpby` takes a moment: docc translates the function into its intermediate representation, parallelizes the loop across all CPU cores with OpenMP, and compiles it to native code. Later calls with arrays of the same type reuse the compiled code and run at native speed.

## Step 2: Offload to a GPU

Change the target to run the same function on a GPU. GPU targets additionally need [CuPy](https://cupy.dev/):

```bash
pip install cupy-cuda12x     # NVIDIA, CUDA 12
pip install cupy-rocm-7-0    # AMD, ROCm 7
```

```python
@native(target="cuda")       # or target="rocm"
def waxpby(alpha, x, beta, y, w):
    for i in range(x.shape[0]):
        w[i] = alpha * x[i] + beta * y[i]
```

docc copies the NumPy arrays to the GPU, runs the kernel, and copies `w` back. It also warns you about these copies, because they often take longer than the computation itself. [Usage](../usage.md#keep-data-on-the-gpu) shows how to keep the data on the GPU.

## Step 3: Write it the NumPy way

You do not have to write loops. docc also understands NumPy expressions and returns new arrays:

```python
@native(target="openmp")
def waxpby(alpha, x, beta, y):
    return alpha * x + beta * y

w = waxpby(2.0, x, 3.0, y)
```

## What happened

On the first call, docc parsed the Python source of `waxpby`, used the types and shapes of the arguments to translate it into its intermediate representation (SDFG), optimized it for the target, and compiled the generated C++ code with Clang into a shared library. The compiled function is cached per argument types, so calling it again with new data does not compile again.

## Next step

To build the package yourself, continue with [Custom Installation](../custom-installation.md). To control how docc compiles WAXPBY, see [Usage](../usage.md).
