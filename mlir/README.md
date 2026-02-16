# MLIR

The MLIR module contains a dialect and translation of core MLIR dialects to SDFGs.

- **SDFG Dialect**: A dialect to represent SDFGs in MLIR.
- **Conversion**: Conversion passes to convert core dialects (arith, linalg, etc.) to the SDFG dialect.
- **Translate**: A translation pass to dump an SDFG from the SDFG dialect.

## Build from Sources

To build the MLIR component from source install LLVM/MLIR dependencies first:
```bash
sudo apt-get install -y libmlir-19 libmlir-19-dev mlir-19-tools
```

Second, build from the root of the repository using cmake:

```bash
mkdir build && cd build
cmake -G Ninja \
    -DCMAKE_C_COMPILER=clang-19 \
    -DCMAKE_CXX_COMPILER=clang++-19 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMLIR_BUILD_FRONTEND=ON \
    -DMLIR_BUILD_TESTS=ON \
    -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-19/build/utils/lit/lit.py \
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
