---
title: Use docc with PyTorch
sidebar_label: Install & Usage
sidebar_position: 1
description: Use docc as a torch.compile backend for inference and training
---

docc registers a `torch.compile` backend. Models are imported from PyTorch, lowered to an optimized SDFG, and executed.

## Install

The PyTorch frontend is published on [PyPI](https://pypi.org/project/docc-ai/):

```bash
pip install docc-ai
```

It uses `torch-mlir` to translate models to core MLIR dialects. Install its requirements from a checkout of the docc repository:

```bash
pip install -r mlir/requirements.txt
```

To build the MLIR component from source, see the [MLIR component README](https://github.com/daisytuner/docc/tree/main/mlir).

## Inference

Use `torch.no_grad()` to enforce inference via the dynamo backend:

```python
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

model = LinearRegression()
example_input = torch.randn(2, 4)

with torch.no_grad():
    compiled_model = torch.compile(model, backend="docc", options={"target": "openmp", "category": "server"})

res = compiled_model(example_input)
```

## Training (experimental)

Training is supported experimentally via an AOTAutograd integration:

```python
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

torch.manual_seed(42)
model = LinearRegression()

program = torch.compile(model, backend="docc", options={"target": "openmp", "category": "server"})
optimizer = torch.optim.SGD(program.parameters(), lr=0.5)
criterion = nn.MSELoss()

for _ in range(20):
    x = torch.randn(32, 2)
    target = x  # identity: output should equal input

    optimizer.zero_grad()
    res = program(x)
    loss = criterion(res, target)
    loss.backward()
    optimizer.step()
```
