---
title: Run Your First Code
sidebar_label: First Program
sidebar_position: 2
description: Compile a vector kernel with docc and run it on CPU cores and GPUs
---

Throughout this guide, we use one running example: **WAXPBY**, the vector update `w = alpha * x + beta * y`. It is plain C, with no pragmas, intrinsics, or GPU code.

```c title="waxpby.c"
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 24)

int main(void) {
    float* x = malloc(N * sizeof(float));
    float* y = malloc(N * sizeof(float));
    float* w = malloc(N * sizeof(float));

    float alpha = 2.0f, beta = 3.0f;
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(N - i);
    }

    // The kernel: w = alpha * x + beta * y
    for (int i = 0; i < N; i++) {
        w[i] = alpha * x[i] + beta * y[i];
    }

    printf("w[0] = %f, w[N-1] = %f\n", w[0], w[N - 1]);
    free(x); free(y); free(w);
    return 0;
}
```

## Step 1: Compile it like with clang

`docc` accepts the same flags as `clang`, so you can use it on existing code right away:

```bash
docc -O3 waxpby.c -o waxpby
./waxpby
```

Without further options, docc behaves like a regular compiler and the program runs sequentially.

## Step 2: Use all CPU cores

Select a target with `-docc-tune`. With `openmp`, docc detects that the loop iterations are independent and parallelizes the loop across all cores:

```bash
docc -O3 -docc-tune=openmp waxpby.c -o waxpby
./waxpby
```

## Step 3: Offload to a GPU

The same source runs on a GPU when you change the target. docc generates the GPU kernel and the data transfers between host and device:

```bash
# NVIDIA
docc -O3 -docc-tune=cuda waxpby.c -o waxpby

# AMD
docc -O3 -docc-tune=rocm waxpby.c -o waxpby
```

During compilation, docc reports which GPU it compiles for, for example:

```text
[DOCC] Compiling CUDA for sm_90 of NVIDIA H100
```

Tenstorrent Wormhole and Blackhole cards work the same way with `-docc-tune=tenstorrent`.

## What happened

docc compiled your program with Clang, lifted the kernel loop into its intermediate representation (SDFG), parallelized it for the selected target, and linked the generated code back into your executable. Code that docc cannot optimize is compiled as usual, so the program always behaves the same as with `clang`.

## Next step

Real applications are rarely a single file. In [Using Libraries](./libraries.md), we move the WAXPBY kernel into a shared library and see how docc avoids unnecessary GPU transfers across library calls.
