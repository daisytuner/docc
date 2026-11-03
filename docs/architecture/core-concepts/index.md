---
title: Core Concepts
sidebar_label: Overview
sidebar_position: 1
description: The building blocks of the docc core - Structured SDFGs, analyses, passes, and transformations
---

The core of docc (`sdfg/` and `opt/`) is independent of any frontend or target. It is built from four concepts:

| Concept                                        | What it is                                                                 | Changes the SDFG? |
| ---------------------------------------------- | -------------------------------------------------------------------------- | :---------------: |
| [Structured SDFG](./structured-sdfgs.md)       | The intermediate representation: data, a tree of control flow, and dataflow graphs | —        |
| [Analyses](./analyses.md)                      | Read the SDFG and compute facts about it, for example loops and data dependencies  | no       |
| [Passes](./passes.md)                          | Rewrite the whole SDFG, for example to simplify it or to schedule all loops        | yes, globally |
| [Transformations](./transformations.md)        | Rewrite one specific place, for example tile one loop, if it is legal              | yes, locally  |

## How they work together

```text
                       ┌──────────────────────────┐
  frontend ──builds──► │     Structured SDFG      │ ──► code generation
                       └──────────────────────────┘
                          ▲         │         ▲
                  rewrite │         │ read    │ rewrite
                          │         ▼         │
                  ┌───────────┐ ┌──────────┐ ┌─────────────────┐
                  │  Passes   │ │ Analyses │ │ Transformations │
                  └───────────┘ └──────────┘ └─────────────────┘
                          │         ▲         │
                          └── query ┴─ query ─┘
```

- Frontends create a Structured SDFG with a **builder** (`StructuredSDFGBuilder`). All changes go through the builder, never directly through the SDFG.
- **Analyses** are computed on demand by an `AnalysisManager` and cached until the SDFG changes.
- **Passes** and **transformations** ask the analysis manager for the facts they need, for example whether a loop has dependencies between iterations, and then change the SDFG through the builder.
- The compilation pipelines of all frontends are sequences of passes; [scheduling](./passes.md#scheduling), the step that maps loops onto a target, is itself a pass that applies transformations.

## Follow along in Python

The `docc-compiler` package exposes the core through Python bindings (`docc.sdfg`). All examples in this section use them, so you can run them after `pip install docc-compiler`. They use the running example of the user guides, **WAXPBY** (`w = alpha * x + beta * y`), built by hand and then analyzed, optimized, and compiled.

To look at the SDFG that a frontend creates instead, use the Python frontend:

```python
import numpy as np
from docc.python import native

@native(target="openmp")
def waxpby(alpha, x, beta, y, w):
    for i in range(x.shape[0]):
        w[i] = alpha * x[i] + beta * y[i]

x = np.ones(1024, dtype=np.float32)
waxpby.compile(2.0, x, 3.0, x, np.zeros_like(x))
sdfg = waxpby.last_sdfg         # docc.sdfg.StructuredSDFG
print(sdfg.to_cpp())            # generated C++ code
```

The C++ API of the core is documented in the [C++ API reference](../../reference/api.md).
