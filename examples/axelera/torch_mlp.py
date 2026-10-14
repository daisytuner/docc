"""A two-layer MLP in PyTorch, compiled for Axelera Europa through torch.compile.

The docc backend exports the model, runs both linear layers on the MVM (their (out, in) weights transposed by the
DWPU after quantization) and the ReLU on DPU datapaths, compiles the generated AxScript with alto, and runs it on
docc's AxScript emulator. The layers have no bias, since adding a bias row does not map onto datapaths yet.

    DOCC_AXELERA_PYTHON=/path/to/axelera-sdk/bin/python python examples/axelera/torch_mlp.py
"""

import os
import sys

import torch

import docc.pytorch  # noqa: F401 (registers the "docc" backend)


def mlp(features):
    return torch.nn.Sequential(
        torch.nn.Linear(features, features, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(features, features, bias=False),
    )


def main(cores=2, size=128):
    if not os.environ.get("DOCC_AXELERA_PYTHON"):
        sys.exit(
            "set DOCC_AXELERA_PYTHON to an interpreter with Axelera's SDK (axelera-compiler) installed"
        )

    torch.manual_seed(0)
    model = mlp(size).eval()
    x = torch.randn(size, size)

    options = {
        "target": "axelera",
        "codegen": "axscript",
        "symbolic_shapes": False,
        "axelera.num_cores": cores,
    }
    with torch.no_grad():
        compiled = torch.compile(model, backend="docc", options=options)
        out = compiled(x)
        expected = model(x)

    error = ((out - expected).abs().max() / expected.abs().max()).item()
    print(f"max |error| / max |expected| = {error:.2%} (int8 per-tensor quantization)")


if __name__ == "__main__":
    main()
