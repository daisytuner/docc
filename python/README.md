# Python Bindings and Frontend

This component provides Python bindings via pybind11 to build SDFGs with Python.
Additionally, this component provides the `@native` decorator, which automatically
converts Python functions into SDFGs and codegens them for the available target.

- **Bindings**: Build SDFGs programmatically with Python
- **AST Parser**: Automatically compile Python/NumPy code to optimized native code
- **Targets**: Support for CPU (sequential, OpenMP, SIMD via Highway), CUDA GPUs, and other accelerators

## Build from Sources

To build the Python component from sources run pip on the component's directory:

```bash
pip install -e python/
```

### Requirements

#### Python Dependencies

- Python >= 3.11
- NumPy >= 1.19.0

#### System Dependencies

The following system dependencies must be installed (same as core components):

```bash
sudo apt-get install -y libgmp-dev libzstd-dev
sudo apt-get install -y nlohmann-json3-dev
sudo apt-get install -y libboost-graph-dev libboost-graph1.74.0
sudo apt-get install -y libisl-dev
```

#### Compiler Requirements

**Clang/LLVM 19 is required** for code generation. Install it with:

```bash
# Ubuntu/Debian
sudo apt-get install -y clang-19 llvm-19
```

For CUDA support, you also need the NVIDIA CUDA Toolkit installed.

## Usage

### The `@native` Decorator

The `@native` decorator is the primary way to use the Python frontend. It automatically:
1. Parses the Python function's AST
2. Converts it to an SDFG representation
3. Applies optimizations based on the target
4. Compiles to native code
5. Executes and returns results

### Basic Example

```python
from docc.python import native
import numpy as np

@native
def vector_add(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]

# Usage
N = 1024
A = np.random.rand(N).astype(np.float64)
B = np.random.rand(N).astype(np.float64)
C = np.zeros(N, dtype=np.float64)

vector_add(A, B, C, N)  # JIT compiles and executes
```

## Compilation Targets

The `@native` decorator accepts a `target` parameter to specify the code generation backend:

### `target="none"` (Default)

No scheduling or optimization is applied. The SDFG is compiled as-is without parallelization. Useful for debugging or when you want to manually control the generated code.

```python
@native(target="none")
def simple_loop(A, B, N):
    for i in range(N):
        B[i] = A[i] * 2.0
```

### `target="sequential"`

Generates optimized sequential code with SIMD vectorization using [Google Highway](https://github.com/google/highway). This target automatically vectorizes eligible loops using portable SIMD intrinsics.

```python
import math

@native(target="sequential")
def vectorized_sin(A, B):
    for i in range(A.shape[0]):
        B[i] = math.sin(A[i])

# Highway automatically vectorizes the sin computation
N = 128
A = np.random.rand(N).astype(np.float64)
B = np.zeros(N, dtype=np.float64)
vectorized_sin(A, B)
```

**Supported Highway operations include:**
- Trigonometric: `sin`, `cos`, `asin`, `acos`, `atan`, `atan2`
- Exponential: `exp`, `log`, `log10`, `log2`, `pow`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Other: `sqrt`, `abs`, `floor`, `ceil`, `round`

### `target="openmp"`

Generates parallel code using OpenMP. Suitable for multi-core CPUs. Loops are automatically parallelized with appropriate scheduling.

```python
@native(target="openmp", category="desktop")
def parallel_add(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]

# Executes in parallel across CPU cores
N = 1000000
A = np.random.rand(N).astype(np.float64)
B = np.random.rand(N).astype(np.float64)
C = np.zeros(N, dtype=np.float64)
parallel_add(A, B, C, N)
```

### `target="cuda"`

Generates CUDA code for NVIDIA GPUs. Loops are mapped to GPU thread blocks and threads.

```python
@native(target="cuda", category="server")
def gpu_add(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]

# Executes on GPU (data is automatically transferred)
N = 1024
A = np.random.rand(N).astype(np.float64)
B = np.random.rand(N).astype(np.float64)
C = np.zeros(N, dtype=np.float64)
gpu_add(A, B, C, N)
```

## The `category` Parameter

The `category` parameter provides hints to the scheduler about the target hardware:

- `"edge"`
- `"desktop"`
- `"server"`

```python
@native(target="openmp", category="desktop")
def cpu_kernel(A, B):
    ...

@native(target="cuda", category="server")
def gpu_kernel(A, B):
    ...
```

## Advanced: Manual Compilation

For more control, you can manually compile and cache the SDFG:

```python
@native(target="openmp")
def my_kernel(A, B, C, N):
    for i in range(N):
        C[i] = A[i] + B[i]

# Pre-compile with sample arguments
compiled = my_kernel.compile(A, B, C, N)

# Reuse compiled version
result = compiled(A, B, C, N)

# Access the underlying SDFG
sdfg = my_kernel.last_sdfg
```

## License

This component is part of docc and is published under the BSD-3-Clause license.
See [LICENSE](../LICENSE) for details.
