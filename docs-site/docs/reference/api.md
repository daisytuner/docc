---
title: C++ API Reference
sidebar_label: C++ API
sidebar_position: 1
description: Doxygen reference for the C++ headers of all docc modules
---

The C++ API reference is generated with [Doxygen](https://www.doxygen.nl/) from the headers of all docc modules.

**[Open the C++ API reference →](pathname:///api/index.html)**

## Covered modules

| Module           | Headers                  |
| ---------------- | ------------------------ |
| `sdfg`           | `sdfg/include`           |
| `opt`            | `opt/include`            |
| `llvm`           | `llvm/include`           |
| `mlir`           | `mlir/include`           |
| `c-compile`      | `c-compile/include`      |
| `rpc`            | `rpc/include`            |
| `rtl`            | `rtl/include`            |
| `arg-capture-io` | `arg-capture-io/include` |
| `targets`        | `targets/et/include`     |

The Python and PyTorch components expose no C++ headers.

## Building locally

From `docs-site/`, regenerate the reference with:

```bash
npm run api
```

This requires `doxygen` and `graphviz`. The output is written to `static/api/` and is not checked in.
