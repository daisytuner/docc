---
title: Debug Options
sidebar_position: 4
description: Inspect what docc does with a PyTorch model, debug generated code, and run the tests
---

This page is for developers who want to understand how docc compiles a model, or who work on docc itself. We continue with ResNet-18 from the [Quickstart](./quickstart/first-program.md).

The backend uses the compiler of the Python package, so the [Python debug options](../python/debugging.md) apply. This page shows how to use them with `torch.compile`.

## Look at the intermediate steps

Set `DOCC_DEBUG=dump` (or the option `debug_dump=True`) to write the SDFG after every compilation stage, together with the generated code:

```bash
DOCC_TMP=./docc-out DOCC_DEBUG=dump python resnet18.py
```

docc creates one directory per compiled graph, for example `docc-out/GraphModule-60ce9405a45dca28/`. It contains the SDFG after each stage (`__docc_GraphModule.py0.parsed` … `py5.post_sched`, as JSON and DOT), the generated C++ code, for GPU targets one `kernel_*.cu` file per GPU kernel, and the compiled library. See [Python Debug Options](../python/debugging.md#look-at-the-intermediate-steps) for the list of stages.

To inspect the SDFG from your script, use the `on_compile` callback:

```python
def inspect(info):
    sdfg = info["sdfg"]                       # docc.sdfg.StructuredSDFG
    print(info["name"], "on GPU:", info["device_resident"])

compiled = torch.compile(model, backend="docc", options={"target": "cuda", "on_compile": inspect})
```

To see what PyTorch hands to docc, use PyTorch's own logging, for example `TORCH_LOGS="graph_code"`.

## Debug the generated code

| Option                                                | Description                                                                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `DOCC_DEBUG=build` / `debug_build=True`               | Compile the generated code with debug information (`-g`).                                                |
| `DOCC_DEBUG=build_threads=N` / `build_thread_count=N` | Number of threads to compile the generated code with.                                                    |
| `DOCC_CUDA_DEBUG=1`, `DOCC_ROCM_DEBUG=1`              | Check every CUDA/HIP call and kernel launch in the generated code and report errors with their location. |
| `target="none"`                                       | Compile without parallelization to check whether a problem is caused by the target.                      |
| `approximate_math=False`                              | Use exact math functions to rule out precision differences from fast GPU math.                           |

To find out whether a wrong result comes from docc, compare the compiled model with eager PyTorch:

```python
with torch.no_grad():
    expected = model(x)
    actual = torch.compile(model, backend="docc", options={"target": "openmp"})(x)
print((actual - expected).abs().max())
```

`DOCC_CUDA_DEBUG` and `DOCC_ROCM_DEBUG` take effect while the code is generated. If the graph is already cached, remove its directory in `DOCC_TMP` first.

## Debug builds and tests

Build the compiler in debug mode as described in [Custom Installation](./custom-installation.md#debug-builds). The tests use pytest and select the target with `--target`:

```bash
pytest -n auto pytorch/tests                       # all tests, target "none"
pytest -n auto pytorch/tests --target=openmp
pytest pytorch/tests/models --target=cuda          # complete models on an NVIDIA GPU
```

| Directory                     | Tests                                         |
| ----------------------------- | --------------------------------------------- |
| `pytorch/tests/ops`           | Individual PyTorch operators                  |
| `pytorch/tests/pipelines`     | Several operators working together            |
| `pytorch/tests/models`        | Complete models such as ResNet and SegFormer  |
| `pytorch/tests/graph_parser`  | Internals of the graph translation            |
