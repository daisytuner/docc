# The PyTorch Frontend

This frontend enables the translation of PyTorch GraphModules into Structured SDFGs.
It replaces the old approach of using [torch-mlir](https://github.com/llvm/torch-mlir) to generate MLIR from PyTorch GraphModules and translating the MLIR to Structured SDFGs via an [MLIR frontend](../mlir/).

## How to Use

The PyTorch frontend is available as a backend for `torch.compile`.
All you have to do to use it with your existing PyTorch model is add `import docc.pytorch` and extend your `torch.compile` call with `backend="docc"`.
You can set all additional options with the `options={...}` dictionary argument of `torch.compile`.
```python
...
import docc.pytorch
...
... = torch.compile(..., backend="docc", ...)
...
```

> [!NOTE]
> Currently, only inference is supported.
> If your model uses AOTAutograd, you have to disable it.
> You can wrap your `torch.compile` in a `torch.no_grad()` context, for example.

## Example

This example shows a simple image-classification task using ResNet18.
The corresponding source file is located in [pytorch/examples/resnet18.py](./examples/resnet18.py).

```python
import torch
import torchvision.models as models
import PIL.Image as Image
import requests

import docc.pytorch

image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"

weights = models.resnet.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

image = Image.open(requests.get(image_url, stream=True).raw)
transforms = weights.transforms()
input_tensor = transforms(image).unsqueeze(0)

with torch.no_grad():
    program = torch.compile(model, backend="docc", options={"target": "openmp"})
    output_tensor = program(input_tensor)

probs = torch.softmax(output_tensor, dim=1)
print(f"Probability of Samoyed (dog): {(probs[0, 258] * 100):.3f}%")
```

The PyTorch model is parallelized on the CPU.
If, for example, you want to execute this model on an NVIDIA GPU and enable remote tuning, adapt the `options` attribute as follows:
```python
    program = torch.compile(model, backend="docc", options={"target": "cuda", "remote_tuning": True})
```

## Installation From Source

> [!NOTE]
> These instructions assume that the libraries required to build sdfglib are already installed. See [here](../sdfg/) for more information.

### 1. Installing dependencies

Create a virtual Python environment and install the dependencies from the requirements (.txt) file that matches your needs.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Use [`requirements.txt`](./requirements.txt) to install PyTorch without GPU support.

```bash
pip install -r pytorch/requirements.txt
```

Use [`requirements-cuda.txt`](./requirements-cuda.txt) to install PyTorch with CUDA support and CuPy.

```bash
pip install -r pytorch/requirements-cuda.txt
```

Use [`requirements-rocm.txt`](./requirements-rocm.txt) to install PyTorch with ROCm support and CuPy (for ROCm).

```bash
pip install -r pytorch/requirements-rocm.txt
```

If you want to use your own PyTorch and CuPy versions because of different drivers, GPU runtimes, or other requirements, use [`requirements-base.txt`](./requirements-base.txt) to install only the base dependencies.

```bash
pip install -r pytorch/requirements-base.txt
```

> [!NOTE]
> This frontend is regularly tested only against the PyTorch and CuPy versions specified in the aforementioned requirements (.txt) files.
> If you encounter any problems, please [open an issue in the docc repository](https://github.com/daisytuner/docc/issues).

### 2. Install Python frontend

Since the PyTorch frontend builds upon the [Python frontend](../python/), we first need to install it.
Because we are building the PyTorch frontend from source, it is recommended to also build the Python frontend from source.

```bash
pip install --no-build-isolation -e python/
```

> [!NOTE]
> To build with debug information and debug output, add `-Ccmake.build-type=Debug`.

> [!TIP]
> To speed up (re-)build times in the future add `-Cbuild-dir=` followed by a folder name.
> For example, `-Cbuild-dir=build` will create the folder `python/build/` where the CMake build artifacts are stored.

### 3. Install the PyTorch frontend

```bash
pip install --no-build-isolation -e pytorch/
```

## Testing

There are many integration tests for the functionality of the PyTorch frontend.
They are written as pytests and located in [pytorch/tests/](./tests/).
The following list gives an overview of the subfolders:
- `graph_parser`: Contains tests for specific graph parser functionality.
- `ops`: Contains a large number of tests for individual PyTorch operations.
- `pipelines`: Contains tests for multiple PyTorch operations working together.
- `models`: Contains tests for complete models.

## Functionality

This frontend is written in Python.
It uses the Python bindings from the [Python frontend](../python/) to construct Structured SDFGs.

The starting point and core of the frontend is `PyTorchProgram`, located in [pytorch/docc/pytorch/pytorch_program.py](./docc/pytorch/pytorch_program.py#L100).
It is registered by a wrapper as a custom PyTorch Dynamo backend (`torch.compile`).
When it receives a PyTorch graph module, it uses `torch.export` to lower it to PyTorch Core ATen IR.
The resulting IR is then parsed by a custom graph parser, which translates it into a Structured SDFG and returns a `CompiledSDFG` (see the Python frontend for more details).

The `GraphParser` class is located in [pytorch/docc/pytorch/graph_parser/__init__.py](./docc/pytorch/graph_parser/__init__.py#L60).
It has a modular registry in which a `GraphParserModule` (located in [pytorch/docc/pytorch/graph_parser/utils.py](./docc/pytorch/graph_parser/utils.py#L1033)) can be registered for each Core ATen IR operation.
