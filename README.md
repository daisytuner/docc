[![Coverage Status](https://coveralls.io/repos/github/daisytuner/docc/badge.svg?branch=main)](https://coveralls.io/github/daisytuner/docc?branch=main)

The Daisytuner Optimizing Compiler Collection (docc) implements an intermediate representation as well as frontends, drivers, and code generation for translation and optimization of various programming languages to multiple targets.

The core of the project is stateful dataflow multigraphs (SDFG), implemented in the `sdfg` module.
It contains the definition of the intermediate representation as well as numerous passes and analyses.
For instance, docc comes with support for auto-parallelization using data-centric and polyhedral analysis.

SDFGs can be generated from Python (JIT) and MLIR frontends, which are separate components including Python bindings and an MLIR dialect for conversion.
Targets such as Generic, [Google Highway](https://github.com/google/highway), [OpenMP](https://www.openmp.org/), and [CUDA](https://developer.nvidia.com/cuda/toolkit) are implemented in `opt`.

Furthermore, the repository contains runtime libraries for code instrumentation (performance counters and data capturing).

## Compatibility

### Frontend / Backend Matrix

|                      | Highway | OpenMP | CUDA | ROCm | Metal |
|----------------------|:-------:|:------:|:----:|:----:|:-----:|
| Python (Linux)       | ✅      | ✅     | ✅   | 🚧   | ❌    |
| Python (macOS)       | ✅      | ✅     | ❌   | ❌   | 🚧    |
| PyTorch (Linux)      | ✅      | ✅     | ✅   | 🚧   | ❌    |
| PyTorch (macOS)      | 🚧      | 🚧     | 🚧   | 🚧   | 🚧    |
<!-- | ONNX (Linux)         | 🚧      | 🚧     | 🚧   | 🚧   | 🚧    |
| ONNX (macOS)         | 🚧      | 🚧     | 🚧   | 🚧   | 🚧    | -->

✅ Supported | ❌ Not supported | 🚧 Work in progress / planned

### Targets

Each target enables a specific combination of backends:

| Target       | Transfer Tuning | Highway | OpenMP | CUDA | Metal |
|--------------|:---------------:|:-------:|:------:|:----:|:-----:|
| `sequential` | ✅              | ✅      | ❌     | ❌   | ❌    |
| `openmp`     | 🚧              | ✅      | ✅     | ❌   | ❌    |
| `cuda`       | 🚧              | ✅      | ✅     | ✅   | ❌    |
<!-- | `metal`      | 🚧              | ✅      | ✅     | ❌   | ✅    | -->

Transfer Tuning refers to a collection of dataflow optimizations using optimization databases.

## Quick Start

Binary releases are published for each new version and can be downloaded via standard package managers.
They provide an easy way to get started with docc.

### Python

The Python frontend generates native C++ code, which is compiled and called from Python.
This requires `clang-19` to be installed on the system (see [LLVM releases](https://apt.llvm.org/)).

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

### MLIR (PyTorch)

The MLIR frontend can be installed from PyPi:

```bash
pip install docc-ai # (soon)
```

To use the frontend with PyTorch, also install `torch-mlir`, which we use to translate models to core MLIR dialects initially:

```bash
pip install --pre torch-mlir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

This allows you to import models directly from PyTorch and generate an optimized SDFG:

```python
import torch
import torch.nn as nn

import docc.torch
docc.torch.set_backend_options(target="openmp", category="server")

class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

model = IdentityNet()
example_input = torch.randn(2, 1)

# Compile model
with torch.no_grad():
  compiled_model = torch.compile(model, backend="docc")

# Forward
res = compiled_model(example_input)
```

For further details, check out the [component's README.md](./mlir/).

## Building the Core Components

```bash
sudo apt-get install -y libgmp-dev libzstd-dev
sudo apt-get install -y nlohmann-json3-dev
sudo apt-get install -y libboost-graph-dev
sudo apt-get install -y libisl-dev
sudo apt-get install -y libcurl4-gnutls-dev
```

The core components `sdfg`, `opt`, `rtl` and `rpc` can be built with cmake.

```bash
mkdir build && cd build
cmake \
  -G Ninja \
  -DCMAKE_C_COMPILER=clang-19 \
  -DCMAKE_CXX_COMPILER=clang++-19 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS:BOOL=OFF \
  -DBUILD_BENCHMARKS:BOOL=OFF \
  -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF  \
  ..
ninja -j$(nproc)
```

For instructions on how to build and extend frontends, check out the README.md of the components' directories.

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
