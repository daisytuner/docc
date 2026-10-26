---
title: Debug Options
sidebar_position: 4
description: Inspect the SDFGs and generated code of @native functions, debug GPU code, and run the tests
---

This page is for developers who want to understand how docc compiles a function, or who work on docc itself. We continue with WAXPBY from the [Quickstart](./quickstart/first-program.md).

## Look at the intermediate steps

Set `DOCC_DEBUG=dump` (or the option `debug_dump=True`) to write the SDFG after every compilation stage, together with the generated code:

```bash
DOCC_TMP=./docc-out DOCC_DEBUG=dump python waxpby.py
```

docc creates one directory per function, target, and argument signature:

```text
docc-out/waxpby.py-waxpby-openmp-ae0ce0211084fb5a/
├── waxpby_sdfg.py0.parsed.json / .dot         translated from Python
├── waxpby_sdfg.py2.expanded.json / .dot       library nodes (e.g. NumPy calls) expanded
├── waxpby_sdfg.py3.1.pre-fusion.json / .dot
├── waxpby_sdfg.py3.2.post-fusion.json / .dot  after loop fusion
├── waxpby_sdfg.py3.opt.json / .dot            after optimization
├── waxpby_sdfg.py4.norm.json / .dot           after loop normalization
├── waxpby_sdfg.py5.post_sched.json / .dot     after scheduling for the target
├── waxpby_sdfg.cpp / .h                       generated C++ code
└── libwaxpby_sdfg.so                          compiled library
```

Render a stage with Graphviz:

```bash
dot -Tsvg docc-out/waxpby.py-waxpby-openmp-*/waxpby_sdfg.py5.post_sched.dot -o waxpby.svg
```

The SDFG of the last compilation is also available in Python, and `compile` accepts a fixed output directory:

```python
kernel = waxpby.compile(2.0, x, 3.0, y, w, output_folder="./waxpby-out")
sdfg = waxpby.last_sdfg       # docc.sdfg.StructuredSDFG
```

## Debug the generated code

| Option                                  | Description                                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `DOCC_DEBUG=build` / `debug_build=True` | Compile the generated code with debug information (`-g`), so you can step through it with `gdb`.  |
| `DOCC_DEBUG=build_threads=N` / `build_thread_count=N` | Number of threads to compile the generated code with.                                 |
| `DOCC_CUDA_DEBUG=1`, `DOCC_ROCM_DEBUG=1` | Check every CUDA/HIP call and kernel launch in the generated code and report errors with their location. |
| `target="none"`                         | Compile without parallelization to check whether a problem is caused by the target.               |
| `capture_args=True`                     | Save the inputs and outputs of each optimized region to compare them with a reference run.        |

Combine flags in `DOCC_DEBUG` with `;` or `:`, for example `DOCC_DEBUG="dump;build"`. Explicit decorator options take precedence over `DOCC_DEBUG`.

`DOCC_CUDA_DEBUG` and `DOCC_ROCM_DEBUG` take effect while the code is generated. If the function is already cached, remove its directory in `DOCC_TMP` first.

## Compiler statistics

Set `DOCC_STATISTICS=1` to print the time spent in each compiler stage and pass, and how often each pipeline iterated.

## Debug builds and tests

Build the package in debug mode as described in [Custom Installation](./custom-installation.md#debug-builds). The tests use pytest:

```bash
pip install pytest pytest-xdist
pytest -n auto python/tests -m unmarked       # all tests that need no special hardware
pytest -n auto python/tests/python/program    # decorator, caching, and compilation
pytest python/tests -m cuda                   # NVIDIA GPU tests (also: rocm, tenstorrent, etsoc)
```
