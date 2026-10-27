---
title: Docker Images
sidebar_position: 5
description: Prebuilt Docker images to build docc from source and to run it on CPUs, GPUs, and accelerators
---

Daisytuner publishes Docker images on [Docker Hub](https://hub.docker.com/u/daisytuner) that contain everything needed to build docc from source, and the runtimes to run the generated code on GPUs and accelerators. They are the same images that docc's CI uses, so a build that works in the container works in CI.

Use them to:

- build docc without installing LLVM and the build dependencies on your machine,
- try a GPU or accelerator target without setting up CUDA, ROCm, or TT-Metalium yourself,
- reproduce a CI failure locally.

## Choose an image

There are two families. **build-env** images contain the compiler toolchain (LLVM/Clang 21, CMake, Ninja, Python) and all libraries docc depends on. **run-env** images extend the Ubuntu 24.04 build-env image with hardware runtimes and [PAPI](https://icl.utk.edu/papi/) for instrumentation.

| Image                                                                                                                              | Contents                                         | Size\*  |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [`docc-build-env-llvm21-ubuntu-24.04`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-ubuntu-24.04)                     | Build environment, Ubuntu 24.04 (amd64, arm64)   | 1 GB    |
| [`docc-build-env-llvm21-ubuntu-24.04-torch`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-ubuntu-24.04-torch)         | Build environment with libtorch 2.7.1 (C++)      | 1.3 GB  |
| [`docc-run-env-llvm21-ubuntu-24.04`](https://hub.docker.com/r/daisytuner/docc-run-env-llvm21-ubuntu-24.04)                         | CUDA 12.8 and ROCm 7.2 (amd64); CUDA 12.8 (arm64) | 16.5 GB |
| [`docc-run-env-llvm21-ubuntu-24.04-cuda12.8`](https://hub.docker.com/r/daisytuner/docc-run-env-llvm21-ubuntu-24.04-cuda12.8)       | CUDA 12.8 only                                   | 6.3 GB  |
| [`docc-run-env-llvm21-ubuntu-24.04-rocm7.2`](https://hub.docker.com/r/daisytuner/docc-run-env-llvm21-ubuntu-24.04-rocm7.2)         | ROCm 7.2 only                                    | 9 GB    |
| [`docc-run-env-llvm21-ubuntu-24.04-tenstorrent`](https://hub.docker.com/r/daisytuner/docc-run-env-llvm21-ubuntu-24.04-tenstorrent) | TT-Metalium 0.63.0                               | 3.1 GB  |
| [`docc-run-env-llvm21-ubuntu-24.04-et`](https://hub.docker.com/r/daisytuner/docc-run-env-llvm21-ubuntu-24.04-et)                   | ETSoC toolchain                                  |         |

\* Compressed download size of the amd64 image.

The build environment is also available for other distributions, which are used to build the release packages: [`-ubuntu-22.04`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-ubuntu-22.04), [`-ubuntu-25.04`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-ubuntu-25.04), [`-ubuntu-26.04`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-ubuntu-26.04), [`-debian-13`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-debian-13), and [`-rhel-10`](https://hub.docker.com/r/daisytuner/docc-build-env-llvm21-rhel-10) (amd64).

### Tags

The CPU architecture is part of the tag:

| Tag                          | Description                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| `latest-amd64`               | Current image for x86-64                                        |
| `latest-arm64`               | Current image for arm64 (only where listed above)               |
| `beta-amd64`, `beta-arm64`   | Pre-release images that are being tested before they become `latest` |

The Tenstorrent image also includes the TT-Metalium version: `latest-0.63.0-amd64`.

## Build docc from source

Mount your checkout of the [docc repository](https://github.com/daisytuner/docc) into the build-env image. Run the container with your user ID, so that the build files in your checkout belong to you and not to `root`:

```bash
cd docc
docker run --rm -it \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/workspace/docc -w /workspace/docc \
    daisytuner/docc-build-env-llvm21-ubuntu-24.04:latest-amd64 bash
```

Inside the container, follow the build steps of the [C/C++ Custom Installation](./c-cpp/custom-installation.md#build):

```bash
cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_FRONTEND=ON
ninja -C build
./build/llvm/driver/docc -O3 -docc-tune=openmp waxpby.c -o waxpby
```

The build directory stays in your checkout, so the next container picks up where the last one stopped.

## Run on GPUs and accelerators

Use a run-env image and give the container access to the device:

```bash
# NVIDIA GPUs (requires the NVIDIA Container Toolkit on the host)
docker run --rm -it --gpus all \
    -v "$PWD":/workspace/docc -w /workspace/docc \
    daisytuner/docc-run-env-llvm21-ubuntu-24.04-cuda12.8:latest-amd64 bash

# AMD GPUs
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$PWD":/workspace/docc -w /workspace/docc \
    daisytuner/docc-run-env-llvm21-ubuntu-24.04-rocm7.2:latest-amd64 bash

# Tenstorrent cards
docker run --rm -it --device=/dev/tenstorrent \
    -v "$PWD":/workspace/docc -w /workspace/docc \
    daisytuner/docc-run-env-llvm21-ubuntu-24.04-tenstorrent:latest-0.63.0-amd64 bash
```

Then build docc as above and compile with `-docc-tune=cuda`, `rocm`, or `tenstorrent`.

See the documentation of [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) for setting up GPU access in Docker.

To read hardware performance counters with [instrumentation](./c-cpp/usage.md#measure-performance), also add `--cap-add=PERFMON`, as docc's CI does.

## Develop the Python and PyTorch frontends

The images contain Python 3.12. Install `python3-venv` and build the Python packages in a virtual environment inside the container. Run the container without `--user` for this, because `apt-get` needs root:

```bash
apt-get update && apt-get install -y python3-venv
python3 -m venv /tmp/venv && source /tmp/venv/bin/activate
pip install -r pytorch/requirements.txt         # or requirements-cuda.txt / requirements-rocm.txt
pip install --no-build-isolation -e python/ -C build-dir=build-python
pip install --no-build-isolation -e pytorch/
```

See the Custom Installation pages of [Python](./python/custom-installation.md) and [PyTorch](./pytorch/custom-installation.md) for details.

## Preset environment variables

The images set the environment variables that docc needs, so the commands above work without further setup:

| Variable                | Value                       | Images                |
| ----------------------- | --------------------------- | --------------------- |
| `DOCC_BACKEND_COMPILER` | `clang++-21`                | all                   |
| `PATH`, `LD_LIBRARY_PATH` | include `/usr/local/cuda` | CUDA images           |
| `__DAISY_PAPI_VERSION`  | `0x07020000` (PAPI 7.2)     | run-env               |
| `PAPI_ROCM_ROOT`        | `/opt/rocm/`                | ROCm images           |
| `TT_METAL_HOME`         | `/tenstorrent/tt-metalium`  | Tenstorrent images    |

## Other images

The Daisytuner Docker Hub account contains further images. Images with `llvm19` in their name, and images without a docc prefix (for example `x64`, `aster`, `docc-v2`), are previous versions or used internally by Daisytuner's infrastructure. Use the LLVM 21 images listed above.
