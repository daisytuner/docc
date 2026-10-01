---
title: Install docc for Python
sidebar_label: Install
sidebar_position: 1
description: Learn how to install docc for Python
---

The docc Python component provides Python bindings via pybind11 to build SDFGs with Python. It also provides the `@native` decorator, which automatically converts Python functions into SDFGs and generates code for the selected target.

`docc` is available via pip as the primary installation method. You can also [build it from source](./../generated/python-build-from-source.md). The source code of the Python bindings is in the [docc repository](https://github.com/daisytuner/docc/tree/main/python).

## Requirements

The Python frontend generates native C++ code, which is compiled and called from Python. This requires `clang-21` to be installed on the system (see [LLVM releases](https://apt.llvm.org/)).

## Install with pip

Install the `docc-compiler` package from [PyPI](https://pypi.org/project/docc-compiler/):

```bash
pip install docc-compiler
```

## Quick example

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

All targets and options of the decorator are described in [Build from Source & Usage](./../generated/python-build-from-source.md).
