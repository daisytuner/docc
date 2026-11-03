---
title: Transformations
sidebar_position: 5
description: Local, legality-checked rewrites of an SDFG - tiling, parallelization, offloading - and how to record and replay them
---

A **transformation** rewrites one specific place in a Structured SDFG, for example one loop. Unlike a [pass](./passes.md), it does not search the SDFG itself: you, a pass, or a scheduler choose where to apply it. Every transformation first checks whether it is legal there, so applying it never changes what the program computes.

## The `Transformation` interface

| Method                                         | Description                                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| `can_be_applied(builder, analysis_manager)`    | Checks the preconditions, using [analyses](./analyses.md), without changing the SDFG |
| `apply(builder, analysis_manager)`             | Performs the rewrite                                                                |
| `try_apply(builder, analysis_manager)`         | `can_be_applied` followed by `apply`; returns whether it was applied                |
| `to_json(j)`, `from_json(builder, j)`          | Serialize the transformation and where it was applied, for record and replay       |

A transformation is constructed with the node it applies to, for example `LoopTiling(loop, 256)`.

## Example: tile and parallelize WAXPBY

Using `build_waxpby()` from [Structured SDFGs](./structured-sdfgs.md#example-waxpby), this script tiles the loop, parallelizes the tiles, runs them with OpenMP, and compiles the result:

```python
import tempfile
import numpy as np
from waxpby_sdfg import build_waxpby
from docc.sdfg import AnalysisManager, LoopTiling, LoopParallelization, OMPTransform
from docc.compiler.compiled_sdfg import CompiledSDFG

builder = build_waxpby()

def apply(transformation):
    analysis_manager = AnalysisManager(builder)
    assert transformation.can_be_applied(builder, analysis_manager)
    transformation.apply(builder, analysis_manager)

def find_loop(indvar):
    return AnalysisManager(builder).loop_analysis().find_loop_by_indvar(indvar)

apply(LoopTiling(find_loop("i"), 256))                 # chunks of 256 iterations
apply(LoopParallelization(find_loop("i_tile0")))       # outer For -> Map
apply(OMPTransform(find_loop("i_tile0")))              # run the Map with OpenMP

sdfg = builder.move()
compiled = CompiledSDFG(sdfg._compile(tempfile.mkdtemp(), "openmp"), sdfg)

N = 100_000
x, y, w = np.ones(N, np.float32), np.ones(N, np.float32), np.zeros(N, np.float32)
compiled(N, np.float32(2.0), x, np.float32(3.0), y, w)
assert np.allclose(w, 5.0)
```

The generated loop nest:

```cpp
#pragma omp parallel for schedule(static) private(t2, t1, i)
for(i_tile0 = 0;i_tile0 < N;i_tile0 = 256 + i_tile0)
    for(i = i_tile0;(i < N && i < 256 + i_tile0);i = 1 + i)
        ...
```

Each step builds on the previous one. `OMPTransform` only accepts a `Map`, so the loop must be parallelized first. `LoopParallelization` in turn only applies if the iterations are independent. For a prefix sum (`w[i] = w[i - 1] + w[i]`), `can_be_applied` returns `False`, because each iteration reads what the previous one wrote.

Create a new `AnalysisManager` after each change, or reuse one and call `invalidate_all()`, because transformations change the SDFG that the analyses describe. After `builder.move()`, the builder is empty; do not use it again.

## Available transformations

| Group           | Transformations                                                                                                                            | Location                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| Loop structure  | `LoopTiling`, `MultiLevelTiling`, `LoopInterchange`, `LoopSkewing`, `LoopDistribute`, `LoopSplit`, `LoopPeeling`, `LoopShift`, `LoopRotate`, `LoopUnitStride`, `TrivialLoopElimination`, `UnrollTransform` | `opt/include/sdfg/transformations/`        |
| Maps            | `MapCollapse`, `MapFusion`, `MapDelinearize`                                                                                               | `opt/include/sdfg/transformations/`        |
| Parallelization | `LoopParallelization` (`For` → `Map`), `OMPTransform` (OpenMP schedule), `StreamK`                                                          | `opt/include/sdfg/parallelization/`, `opt/include/sdfg/transformations/` |
| Vectorization   | `VectorizeTransform`, `TileVectorizer`                                                                                                     | `opt/include/sdfg/transformations/`, `opt/include/sdfg/tiles/` |
| Tiles           | `TileFusion`, `LocalStorage`, `SoftwarePipelining`                                                                                         | `opt/include/sdfg/tiles/transformations/` |
| GPU             | `CUDATransform`, `CUDAOffloadTransform`, `ROCMOffloadTransform`, `GpuMmaTransform`                                                         | `opt/include/sdfg/transformations/offloading/`, `opt/include/sdfg/targets/` |

All of them except `MultiLevelTiling`, `LoopSplit`, `LoopShift`, `LoopRotate`, `LoopUnitStride`, and `MapDelinearize` are available in Python under the same names in `docc.sdfg`. The target [schedulers](./passes.md#scheduling) are built from these transformations: the OpenMP scheduler, for example, applies `OMPTransform`.

## Record and replay

A `Recorder` applies transformations and records each one, with its parameters and the loop it was applied to, as JSON. A `Replayer` applies a recorded sequence to another SDFG, for example the same function in a later run:

```python
import json
from docc.sdfg import Recorder, Replayer

builder = build_waxpby()                        # a fresh SDFG; the previous builder was moved
recorder = Recorder()
recorder.apply(LoopTiling(find_loop("i"), 256), builder, AnalysisManager(builder))
# ... further transformations

history = recorder.history                      # JSON string
print([step["transformation_type"] for step in json.loads(history)])

other = build_waxpby()
Replayer().apply(other, AnalysisManager(other), history)
```

Recording is how optimizations are exchanged with the Daisytuner service: [transfer tuning](../../c-cpp/usage.md#transfer-tuning) looks up recorded transformation sequences for similar loops and replays them.

## Write a transformation

A transformation derives from `transformations::Transformation`:

```cpp
#include "sdfg/transformations/transformation.h"

class MyTransformation : public sdfg::transformations::Transformation {
    sdfg::structured_control_flow::StructuredLoop& loop_;

public:
    explicit MyTransformation(sdfg::structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

    std::string name() const override { return "MyTransformation"; }

    bool can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder,
                        sdfg::analysis::AnalysisManager& analysis_manager) override {
        // Check preconditions with analyses; do not modify the SDFG here.
        return true;
    }

    void apply(sdfg::builder::StructuredSDFGBuilder& builder,
               sdfg::analysis::AnalysisManager& analysis_manager) override {
        // Rewrite the SDFG through the builder.
    }

    void to_json(nlohmann::json& j) const override {
        j["transformation_type"] = name();
        j["subgraph"]["0"]["element_id"] = loop_.element_id();
    }
};
```

To make it replayable, add a static `from_json(builder, j)` that finds the node with `builder.find_element_by_id(...)` and returns the transformation, and register the type in `dispatch_transformation` in `opt/include/sdfg/transformations/transformation_registry.h`. Registration checks at compile time that `from_json` and `to_json` have the required form. `LoopInterchange` shows the complete pattern.
