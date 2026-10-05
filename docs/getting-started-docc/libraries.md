---
title: Using Libraries
sidebar_position: 3
description: Learn how to compile and link against offloaded libraries with docc
---

Real-world software is rarely just one file. Usually, you have a main application that uses functions from external libraries. In this tutorial, we take the previous example and split it up: the math calculation goes into a shared library, and the main program calls it. The full example code is available on [GitHub](https://github.com/daisytuner/examples/tree/main/example_03).

This shows a special feature of `docc`. Usually, splitting code into libraries makes it hard for compilers to optimize, especially when moving data to an accelerator like a GPU. `docc` can look across these boundaries to keep your code fast.

We need three files. First, a header file `example_03_lib.h` to declare our function:

```c
#ifndef EXAMPLE_03_LIB_H
#define EXAMPLE_03_LIB_H

void waxpby(float alpha, float* x, float beta, float* y, float* w, int n);

#endif
```

Next, the library implementation `example_03_lib.c`. This is where the actual work happens. We want to run this `waxpby` function on an accelerator:

```c
#include "example_03_lib.h"

void waxpby(float alpha, float* x, float beta, float* y, float* w, int n) {
    for (int i = 0; i < n; i++) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}
```

Finally, the main application `example_03_app.c`. It sets up the data and calls the library function twice. Notice how the result `w` from the first call is used right away in the second call:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "example_03_lib.h"

#define N 8194

int main(int argc, char** argv) {
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* w = (float*)malloc(N * sizeof(float));
    float* v = (float*)malloc(N * sizeof(float));

    // Initialize arrays
    float alpha = 2.0f;
    float beta = 3.0f;
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(N - i);
        w[i] = 0.0f;
        v[i] = 0.0f;
    }

    double start = omp_get_wtime();

    // Perform waxpby operation
    waxpby(alpha, x, beta, y, w, N);
    waxpby(alpha, w, beta, y, v, N);

    double end = omp_get_wtime();

    // Print the result
    for (int i = 0; i < 32; i++) {
        printf("v[%d] = %f, ", i, v[i]);
    }
    printf("\n");

    free(x);
    free(y);
    free(w);
    free(v);

    return 0;
}
```

## Building and running

Building is a two-step process. First, compile the library with `docc`, telling it to prepare for CUDA offloading. This creates a shared library (`.so`):

```bash
docc -g -O3 -fPIC -docc-tune=cuda -shared example_03_lib.c -o libExample_03.so
```

Then compile the main application and link it against the library. `-L.` tells the linker to look in the current folder:

```bash
docc -g -O3 -docc-tune=cuda example_03_app.c -lm -L. -lExample_03 -o example_03_app.out
```

To run it, make sure the system can find the new library:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
./example_03_app.out
```

## Why this matters

Normally, separating code like this creates a performance problem. The compiler sees the first function call, copies data to the GPU, does the math, and copies the result back. For the second call, it has to copy that same result to the GPU again. That's a lot of time wasted moving data back and forth.

When `docc` builds the library, it creates several versions of the function for different situations. When it builds the application, it sees how you use the library and picks the best versions.

In this example, `docc` figures out that `w` is created on the GPU and needed there again immediately. It keeps `w` on the GPU and skips the unnecessary trips to the CPU. You get clean, modular code without losing performance.
