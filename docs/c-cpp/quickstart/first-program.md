---
title: Run Your First Code
sidebar_position: 2
description: Learn how to cross-compile a simple example for different processors
---

This guide demonstrates how to offload a simple vector addition kernel to different backends like CUDA and Tenstorrent using `docc`. The example code is written in standard C and can be found on [GitHub](https://github.com/daisytuner/examples/blob/main/example_01/example_01.c).

The following C code implements a simple vector addition (WAXPBY) operation.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 8194

int main(int argc, char** argv) {
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* w = (float*)malloc(N * sizeof(float));

    // Initialize arrays
    float alpha = 2.0f;
    float beta = 3.0f;
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(N - i);
        w[i] = 0.0f;
    }

    double start = omp_get_wtime();

    // Perform waxpby operation: w = alpha * x + beta * y
    for (int i = 0; i < N; i++) {
        w[i] = alpha * x[i] + beta * y[i];
    }

    double end = omp_get_wtime();

    // Print the result
    for (int i = 0; i < 32; i++) {
        printf("w[%d] = %f, ", i, w[i]);
    }
    printf("\n");

    free(x);
    free(y);
    free(w);

    return 0;
}
```

To compile this example with `docc`, use the following command, which is equivalent to standard `clang` or `gcc`:

```bash
docc -g -O3 example_01.c -o example_01.out
./example_01.out
```

`docc` can automatically parallelize your code for multi-core CPUs. To enable this, use the OpenMP tuning mode:

```bash
docc -g -O3 -docc-tune=openmp example_01.c -o example_01.out
```

## Cross-compiling for CUDA and Tenstorrent

Running code on accelerators like NVIDIA GPUs or Tenstorrent devices typically requires rewriting kernels in CUDA or using specialized APIs. With `docc`, you achieve this by changing a single compiler flag.

**CUDA backend:**

```bash
docc -g -O3 -docc-tune=cuda example_01.c -o example_01.out
```

**Tenstorrent backend:**

```bash
docc -g -O3 -docc-tune=tenstorrent example_01.c -o example_01.out
```

All available targets are listed in [Targets](../using-docc/targets.md). Continue with [Using Libraries](./libraries.md).
