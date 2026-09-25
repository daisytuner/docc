---
title: Install docc for C/C++
sidebar_label: Install
sidebar_position: 1
description: Learn how to install docc
---

The Daisytuner Optimizing Compiler Collection (`docc`) is a drop-in replacement for `clang` and `gcc`. It enables you to cross-compile C/C++ applications for different processors and accelerators by simply providing additional compiler flags.

`docc` is available for Linux distributions as a `.deb` package. Follow the instructions below for your operating system.

## Debian / Ubuntu

First, install LLVM 21 from the [official LLVM apt repository](https://apt.llvm.org/) to avoid conflicts with the default package of your distribution.

Next, download the latest release from the [Daisytuner website](https://daisytuner.com/) and install the downloaded package:

```bash
sudo apt-get install ./docc_amd64.deb
```

Continue with [Run Your First Code](./first-program.md).
