---
title: Using Libraries
sidebar_position: 3
description: Move the running example into a shared library and keep data on the GPU across calls
---

Real applications are rarely a single file. Here, we move the WAXPBY kernel from [Run Your First Code](./first-program.md) into a shared library and call it twice from an application.

## The code

The library exposes the kernel:

```c title="waxpby.h"
#pragma once
void waxpby(float alpha, float* x, float beta, float* y, float* w, int n);
```

```c title="waxpby_lib.c"
#include "waxpby.h"

void waxpby(float alpha, float* x, float beta, float* y, float* w, int n) {
    for (int i = 0; i < n; i++) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}
```

The application calls it twice. The result `w` of the first call is the input of the second:

```c title="app.c"
#include <stdio.h>
#include <stdlib.h>
#include "waxpby.h"

#define N (1 << 24)

int main(void) {
    float* x = malloc(N * sizeof(float));
    float* y = malloc(N * sizeof(float));
    float* w = malloc(N * sizeof(float));
    float* v = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
        y[i] = (float)(N - i);
    }

    waxpby(2.0f, x, 3.0f, y, w, N);  // w = 2x + 3y
    waxpby(2.0f, w, 3.0f, y, v, N);  // v = 2w + 3y

    printf("v[0] = %f\n", v[0]);
    free(x); free(y); free(w); free(v);
    return 0;
}
```

## Build and run

Compile both the library and the application with docc and the same target:

```bash
docc -O3 -fPIC -shared -docc-tune=cuda waxpby_lib.c -o libwaxpby.so
docc -O3 -docc-tune=cuda app.c -L. -lwaxpby -o app

LD_LIBRARY_PATH=. ./app
```

## What happened

A conventional compiler treats each library call separately: copy `x` and `y` to the GPU, compute `w`, copy `w` back, and then copy `w` to the GPU again for the second call.

docc looks across the library boundary. When it builds the library, it generates several variants of `waxpby` for different data locations. When it links the application, it sees that `w` is produced on the GPU and consumed there again, so it picks the variants that keep `w` on the device. You get modular code without paying for unnecessary transfers.

:::tip
docc can only optimize across calls into libraries that were also compiled with docc. Other libraries are linked as usual.
:::

## Next step

You now know the basics. To build docc yourself, continue with [Custom Installation](../custom-installation.md). To control how docc optimizes WAXPBY, see [Usage](../usage.md).
