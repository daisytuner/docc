---
title: Usage
sidebar_position: 3
description: Options of the docc torch.compile backend, GPU execution, and environment variables
---

This page explains how to control the docc backend, using the ResNet-18 example from the [Quickstart](./quickstart/first-program.md).

## Compile a model

Import `docc.pytorch` once to register the backend, then pass options as a dictionary to `torch.compile`:

```python
import torch
import docc.pytorch

with torch.no_grad():
    compiled = torch.compile(
        model,
        backend="docc",
        options={"target": "openmp", "category": "desktop"},
    )
    y = compiled(x)
```

docc compiles when `torch.compile` hands it a graph, which happens on the first call. A new input shape or dtype can lead to a new graph that is compiled again.

docc supports **inference only**. With gradients enabled, compilation fails with `BackendCompilerFailed: ... Currently AOTautograd is not supported`.

## Choose a target

| Target       | Hardware        | Description                                                     |
| ------------ | --------------- | --------------------------------------------------------------- |
| `none`       | single CPU core | Default. Compiles the model without parallelization.            |
| `sequential` | single CPU core | Optimizes and vectorizes the code.                              |
| `openmp`     | all CPU cores   | Parallelizes the model with OpenMP.                             |
| `cuda`       | NVIDIA GPUs     | Runs the model on the GPU. Requires CuPy.                       |
| `rocm`       | AMD GPUs        | Runs the model on the GPU. Requires CuPy.                       |

`category` tells the scheduler which class of hardware to optimize for: `edge`, `desktop`, or `server` (default).

## Run on a GPU

For GPU targets, move the model and its inputs to the GPU (`.to("cuda")`, also on AMD GPUs). The compiled model then reads and writes GPU tensors directly, and its outputs stay on the GPU.

You can also leave the model and its inputs on the CPU. docc then copies the data to the GPU and back on every call, warns about it with a `DoccPerformanceWarning`, and returns CPU tensors. As in plain PyTorch, the model and its inputs must be on the same device.

If a model cannot run entirely on the GPU, docc runs it on the host. In that case, keep the model and inputs on the CPU; GPU tensors are rejected with an error that explains this.

## Options

The backend accepts the same options as the [Python `@native` decorator](../python/usage.md#options), for example `target`, `category`, `remote_tuning`, `contraction_precision`, and `pass_options`. Differences:

| Option             | Default | Description                                                                                                  |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------------ |
| `approximate_math` | `True`  | Fast, approximate math functions on GPUs, like the default `torch.compile` backend (Inductor)                 |
| `on_compile`       | —       | Function that docc calls after compiling a graph, see below                                                  |

Unknown options are ignored with a warning.

### Get notified about compilations

`on_compile` receives a dictionary with information about each compiled graph:

```python
def report(info):
    print(info["name"], info["target"], "on GPU:", info["device_resident"])

compiled = torch.compile(model, backend="docc", options={"target": "cuda", "on_compile": report})
```

| Key                | Content                                                       |
| ------------------ | ------------------------------------------------------------- |
| `name`             | Name of the compiled graph                                    |
| `target`, `category` | Options used                                                |
| `device_resident`  | `True` if the model runs entirely on the GPU                  |
| `device_backend`   | `"cuda"`, `"rocm"`, or `None`                                 |
| `sdfg`             | The compiled SDFG (`docc.sdfg.StructuredSDFG`)                |

## Supported models

docc translates the PyTorch core operators ([Core ATen IR](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html)) into its intermediate representation. Common layers such as linear, convolution, pooling, padding, normalization, activations, softmax, reductions, and elementwise operations are supported, and models such as ResNet and SegFormer are tested end to end. The [operator tests](https://github.com/daisytuner/docc/tree/main/pytorch/tests/ops) show the full list. If a model uses an operator or data type that docc does not support yet (for example complex numbers), compilation fails with a `GraphParserError` that names it, wrapped in PyTorch's `BackendCompilerFailed`.

## Environment variables

The backend uses the same environment variables as the Python package, for example `DOCC_TMP`, `DOCC_CUDA_ARCH`, and `DOCC_REUSE_BINARIES`. See [Python environment variables](../python/usage.md#environment-variables).

## Next step

If docc does not compile your model as expected, [Debug Options](./debugging.md) shows how to look at what docc does with ResNet-18.
