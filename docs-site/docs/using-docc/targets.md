---
title: Targets
sidebar_position: 2
description: Which backends each docc target enables and how to select them
---

Each target enables a specific combination of backends:

| Target       | Transfer Tuning | OpenMP | CUDA | ROCm | Metal |
| ------------ | :-------------: | :----: | :--: | :--: | :---: |
| `sequential` |       ✅        |   —    |  —   |  —   |   —   |
| `openmp`     |       🚧        |   ✅   |  —   |  —   |   —   |
| `cuda`       |       🚧        |   ✅   |  ✅  |  —   |   —   |
| `rocm`       |       🚧        |   ✅   |  —   |  ✅  |   —   |

✅ Supported | 🚧 Work in progress

**Transfer tuning** refers to a collection of dataflow optimizations using optimization databases, enabled with [`-docc-transfer-tune`](./commandline.md#-docc-transfer-tune).

The C/C++ compiler additionally supports `tenstorrent` for Tenstorrent Wormhole and Blackhole cards, see [`-docc-tune`](./commandline.md#-docc-tune).

## Selecting a target

| Frontend | How to select                                                        |
| -------- | -------------------------------------------------------------------- |
| C/C++    | `docc -docc-tune=openmp ...`                                         |
| Python   | `@native(target="openmp")`                                           |
| PyTorch  | `torch.compile(model, backend="docc", options={"target": "openmp"})` |

To implement your own target, follow [Adding Custom Offload Targets](./../generated/custom-offload-targets.md).
