"""A two-layer MLP in PyTorch, compiled for Axelera Europa through torch.compile.

The docc backend exports the model, runs both matmuls on the MVM and the ReLU on DPU datapaths, compiles the
generated AxScript with alto, and runs it on docc's AxScript emulator. The weights are stored (in, out) and applied
as `x @ w`, since the transposed weights of nn.Linear do not map onto the MVM yet.

    DOCC_AXELERA_PYTHON=/path/to/axelera-sdk/bin/python python examples/axelera/torch_mlp.py
"""

import os
import sys

import torch

import docc.pytorch  # noqa: F401 (registers the "docc" backend)


class MLP(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(features, features) / features**0.5)
        self.w2 = torch.nn.Parameter(torch.randn(features, features) / features**0.5)

    def forward(self, x):
        return torch.relu(x @ self.w1) @ self.w2


def main(cores=2, size=128):
    if not os.environ.get("DOCC_AXELERA_PYTHON"):
        sys.exit(
            "set DOCC_AXELERA_PYTHON to an interpreter with Axelera's SDK (axelera-compiler) installed"
        )

    torch.manual_seed(0)
    model = MLP(size).eval()
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
