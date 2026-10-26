[![Coverage Status](https://coveralls.io/repos/github/daisytuner/docc/badge.svg?branch=main)](https://coveralls.io/github/daisytuner/docc?branch=main)

The Daisytuner Optimizing Compiler Collection (docc) implements an intermediate representation as well as frontends, drivers, and code generation for translation and optimization of various programming languages to multiple targets.

The core of the project is stateful dataflow multigraphs (SDFG), implemented in the `sdfg` module.
It contains the definition of the intermediate representation as well as numerous passes and analyses.
For instance, docc comes with support for auto-parallelization using data-centric and polyhedral analysis.

SDFGs can be generated from C/C++ (via LLVM), Python (JIT), and PyTorch (`torch.compile`) frontends.
Targets such as Generic, [OpenMP](https://www.openmp.org/), [CUDA](https://developer.nvidia.com/cuda/toolkit), and [ROCm](https://rocmdocs.amd.com/en/latest/) are implemented in `opt`.

Furthermore, the repository contains runtime libraries for code instrumentation (performance counters and data capturing).

The full documentation is available at [docs.daisytuner.com/docc](https://docs.daisytuner.com/docc).

## Compatibility

### Frontend / Backend Matrix

|                      | OpenMP | CUDA | ROCm | Metal |
|----------------------|:------:|:----:|:----:|:-----:|
| C/C++ (Linux)        | ✅     | ✅   | ✅   | —     |
| Python (Linux)       | ✅     | ✅   | ✅  | —     |
| PyTorch (Linux)      | ✅     | ✅   | ✅  | —     |
| Python (macOS)       | ✅     | —    | —    | 🚧   |
| PyTorch (macOS)      | 🚧     | —    | —    | 🚧   |

✅ Supported | 🚧 Work in progress

### Targets

Each target enables a specific combination of backends:

| Target       | Transfer Tuning | OpenMP | CUDA | ROCm | Metal |
|--------------|:---------------:|:------:|:----:|:----:|:-----:|
| `sequential` | ✅              | —      | —    | —    | —     |
| `openmp`     | 🚧              | ✅     | —    | —    | —     |
| `cuda`       | 🚧              | ✅     | ✅   | —    | —     |
| `rocm`       | 🚧              | ✅     | —    | ✅   | —     |

Transfer Tuning refers to a collection of dataflow optimizations using optimization databases.

## Quick Start

Binary releases are published for each new version and can be downloaded via standard package managers.
They provide an easy way to get started with docc.

### Python

The Python frontend generates native C++ code, which is compiled and called from Python.
This requires `clang-21` to be installed on the system (see [LLVM releases](https://apt.llvm.org/)).

Afterwards, simply install docc via [PyPI](https://pypi.org/project/docc-compiler/):

```bash
pip install docc-compiler
```

```python
import numpy as np

from docc.python import native

@native(target="openmp")
def matrix_multiply(A, B):
    return A @ B

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
C = matrix_multiply(A, B)
```

For further details, check out the [component's README.md](./python/).

### PyTorch

The PyTorch frontend is a backend for `torch.compile`. Install PyTorch for your hardware, then the docc compiler and the backend:

```bash
pip install docc-compiler docc-pytorch
```

Import `docc.pytorch` to register the backend and compile your model with `backend="docc"`.
Only inference is supported, so run the compiled model inside `torch.no_grad()`:

```python
import torch
import torchvision.models as models

import docc.pytorch

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    compiled = torch.compile(model, backend="docc", options={"target": "openmp"})
    y = compiled(x)
```

For further details, check out the [component's README.md](./pytorch/).

### C/C++

The C/C++ frontend provides a compiler that can automatically detect parallelism in your code and generate optimized code for multiple targets.
Simply compile your existing C/C++ code with `docc` instead of `clang` or `gcc`, and optionally enable optimizations for specific targets.

To use the C/C++ compiler, install `docc` (Ubuntu 24.04, see [our website](https://daisytuner.com/) for more distros):

```bash
wget -qO docc.deb 'https://firebasestorage.googleapis.com/v0/b/daisy-367210.appspot.com/o/docc-distributables%2Fpackage%2Fdocc_0.4.0-ubuntu_24.04_amd64.deb?alt=media&token=2c32130b-bdfc-41c5-86dd-d3f6de179113'
sudo apt-get install ./docc.deb
```

Then use the `docc` compiler directly on your C/C++ files. Here's a simple example that performs vector addition:

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

Compile this example with `docc` using standard compiler flags:

```bash
docc -g -O3 example.c -o example.out
./example.out
```

To automatically parallelize your code for multi-core CPUs using OpenMP, enable the OpenMP tuning mode:

```bash
docc -g -O3 -docc-tune=openmp example.c -o example.out
```

You can also cross-compile for GPU accelerators. For CUDA:

```bash
docc -g -O3 -docc-tune=cuda example.c -o example.out
```

For further details, check out the [the documentation](https://docs.daisytuner.com/getting-started-docc/install).

## Attribution

docc is based on the specification described in this [paper](https://www.arxiv.org/abs/1902.10345) and the [DaCe reference implementation](https://github.com/spcl/dace).
The license of the reference implementation is included in the `licenses/` folder.

If you use docc, cite the dace paper:
```bibtex
@inproceedings{dace,
  author    = {Ben-Nun, Tal and de~Fine~Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
  title     = {Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures},
  year      = {2019},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series = {SC '19}
}
```

## License

docc is published under the new BSD license, see [LICENSE](LICENSE).
