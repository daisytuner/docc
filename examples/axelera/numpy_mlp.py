"""A two-layer MLP written in numpy, compiled for Axelera Europa.

docc runs both matmuls on the MVM (int8 weights resident in the IMC, row tiles of the activations pipelined through
L1 and split across AI cores) and everything else on DPU datapaths, emits AxScript, compiles it with alto, and runs
it on docc's AxScript emulator.

    DOCC_AXELERA_PYTHON=/path/to/axelera-sdk/bin/python python examples/axelera/numpy_mlp.py
"""

import os
import sys

import numpy as np

from docc.python import native


def mlp(x, w1, w2):
    return np.maximum(x @ w1, 0.0) @ w2


def main(cores=2, size=128):
    if not os.environ.get("DOCC_AXELERA_PYTHON"):
        sys.exit(
            "set DOCC_AXELERA_PYTHON to an interpreter with Axelera's SDK (axelera-compiler) installed"
        )

    rng = np.random.default_rng(0)
    x = rng.standard_normal((size, size), dtype=np.float32)
    w1 = rng.standard_normal((size, size), dtype=np.float32) / np.sqrt(size)
    w2 = rng.standard_normal((size, size), dtype=np.float32) / np.sqrt(size)

    program = native(
        mlp,
        target="axelera",
        codegen="axscript",
        symbolic_shapes=False,
        **{"axelera.num_cores": cores},
    )
    compiled = program.compile(x, w1, w2)
    out = compiled(x, w1, w2)

    expected = mlp(x, w1, w2)
    error = np.abs(out - expected).max() / np.abs(expected).max()
    print(f"AxScript module: {compiled.lib_path}")
    print(
        f"alto output:     {os.path.join(os.path.dirname(compiled.lib_path), 'alto')}"
    )
    print(f"max |error| / max |expected| = {error:.2%} (int8 per-tensor quantization)")


if __name__ == "__main__":
    main()
