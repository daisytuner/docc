# MLIR

> [!CAUTION]
> The MLIR frontend was used as a PyTorch frontend together with [torch-mlir](https://github.com/llvm/torch-mlir).
> However, there is a [PyTorch frontend](../pytorch/) now that directly translates PyTorch GraphModules to SDFGs.

The MLIR frontend contains a translation of core MLIR dialects to SDFGs.
It also contains a custom linalg dialect extension to represent operations that are missing in the core linalg dialect of MLIR.

## Build from Sources

To build the MLIR component from source install LLVM/MLIR dependencies first:
```bash
sudo apt-get install -y libmlir-21 libmlir-21-dev mlir-21-tools
```

Second, build from the root of the repository using cmake:

```bash
mkdir build && cd build
cmake -G Ninja \
    -DCMAKE_C_COMPILER=clang-21 \
    -DCMAKE_CXX_COMPILER=clang++-21 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMLIR_BUILD_FRONTEND=ON \
    -DMLIR_BUILD_TESTS=ON \
    -DMLIR_DIR=/usr/lib/llvm-21/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-21/build/utils/lit/lit.py \
    -DWITH_SYMENGINE_THREAD_SAFE=ON \
    -DWITH_SYMENGINE_RCP=ON \
    -DINSTALL_GTEST=OFF \
    -DBUILD_TESTS:BOOL=OFF \
    -DBUILD_BENCHMARKS:BOOL=OFF \
    -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF \
    ..
ninja -j$(nproc)
```

To verify, the basic dialects and translation work, run the tests:

```bash
ninja check-docc-mlir
```

## License

This component is part of docc and is published under the BSD-3-Clause license.
See [LICENSE](../LICENSE) for details.
