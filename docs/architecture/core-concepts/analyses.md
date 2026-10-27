---
title: Analyses
sidebar_position: 3
description: How docc computes facts about an SDFG, and the analyses that passes and transformations rely on
---

An **analysis** reads a Structured SDFG and computes facts about it: which loops exist, which containers are read and written where, whether iterations of a loop depend on each other. Analyses never change the SDFG. [Passes](./passes.md) and [transformations](./transformations.md) use them to decide what they can change.

## The analysis manager

Analyses are not run eagerly. An `AnalysisManager` creates an analysis the first time it is requested, caches the result, and returns the cached result on later requests:

```cpp
analysis::AnalysisManager analysis_manager(builder.subject());
auto& loops = analysis_manager.get<analysis::LoopAnalysis>();   // computed now
auto& again = analysis_manager.get<analysis::LoopAnalysis>();   // cached
```

When a pass or transformation changes the SDFG, cached analyses become outdated. Passes report which analyses they invalidate (`Pass::invalidates`), and the manager either asks each analysis to update itself for the changed subtree (`Analysis::invalidate`) or discards it, so it is recomputed on the next request. In Python, create a new `AnalysisManager` after a change, or call `invalidate_all()`.

## Example: the WAXPBY loop

Using `build_waxpby()` from [Structured SDFGs](./structured-sdfgs.md#example-waxpby):

```python
from waxpby_sdfg import build_waxpby
from docc.sdfg import AnalysisManager

builder = build_waxpby()
analysis_manager = AnalysisManager(builder)

loops = analysis_manager.loop_analysis()
loop = loops.find_loop_by_indvar("i")
info = loops.loop_info(loop)

print(loop)                        # <For indvar='i' init=0 condition=i < N id=1>
print(info.num_loops)              # 1
print(info.is_perfectly_nested)    # True
print(info.is_perfectly_parallel)  # False: the loop is a For, not yet a Map
```

`is_perfectly_parallel` describes the structure: it is `True` if the loop nest consists only of `Map`s. Whether the `For` *could* become a `Map` is a different question, answered by the dependency analyses below. They show that WAXPBY's iterations are independent, which is why [`LoopParallelization`](./transformations.md) can turn it into a `Map`.

## Available analyses

| Analysis                          | Location | Computes                                                                                            |
| --------------------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| `LoopAnalysis`                    | `sdfg/`  | The loop tree: all loops, their nesting, outermost loops, and per-loop facts (`LoopInfo`)            |
| `Users`                           | `sdfg/`  | Every read, write, and view of every container, with the control flow node where it happens          |
| `DataDependencyAnalysis`          | `sdfg/`  | Which writes each read can observe (read-after-write relations)                                    |
| `LoopCarriedDependencyAnalysis`   | `opt/`   | Dependencies between iterations of a loop; decides whether a loop can be parallelized or is a reduction |
| `MemoryLayoutAnalysis`            | `sdfg/`  | The memory layout (shape and strides) behind each memlet, for example for linearized arrays          |
| `AssumptionsAnalysis`             | `sdfg/`  | Known bounds of symbols at each point, for example `0 <= i < N` inside the loop                     |
| `TypeAnalysis`                    | `sdfg/`  | The element types of containers, including pointers whose type is only known from their use        |
| `ArgumentsAnalysis`               | `sdfg/`  | How each argument of the function is used: read, written, or both                                   |
| `ControlFlowAnalysis`             | `sdfg/`  | Dominance and order between control flow nodes                                                      |
| `ReferenceAnalysis`               | `sdfg/`  | Pointers that alias other containers                                                                |
| `TileAnalysis`, `ReductionBufferAnalysis` | `opt/` | Tiles and reduction buffers in loop nests, used by tiling and GPU code generation |
| `DataTransferEliminationAnalysis` | `opt/`   | Which host-device transfers are redundant                                                           |

The headers are in `sdfg/include/sdfg/analysis/` and `opt/include/sdfg/*/analysis/`; see the [C++ API reference](../../reference/api.md). The Python bindings expose `loop_analysis()`, `users()`, `assumptions_analysis()`, `type_analysis()`, `arguments_analysis()`, `control_flow_analysis()`, and `tile_analysis()`.

## How analyses build on each other

Analyses request other analyses through the manager. Deciding whether a loop can be parallelized, for example, combines several of them:

```text
LoopParallelization.can_be_applied
└── AutoParallelization::classify       → None | Map | Reduce
    ├── LoopAnalysis                    which loop, and what is nested inside
    ├── Users                           which containers the loop body reads and writes
    ├── MemoryLayoutAnalysis            which elements each access touches
    └── LoopCarriedDependencyAnalysis   do two iterations touch the same element?
        └── AssumptionsAnalysis         bounds of the induction variable
```

## Write an analysis

An analysis is a C++ class that derives from `analysis::Analysis` and implements `run`. It can request other analyses from the manager:

```cpp
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"

class CountMapsAnalysis : public sdfg::analysis::Analysis {
    size_t num_maps_ = 0;

protected:
    void run(sdfg::analysis::AnalysisManager& analysis_manager) override {
        auto& loops = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
        for (auto* loop : loops.loops()) {
            if (dynamic_cast<sdfg::structured_control_flow::Map*>(loop)) num_maps_++;
        }
    }

public:
    using Analysis::Analysis;
    std::string name() const override { return "CountMapsAnalysis"; }
    size_t num_maps() const { return num_maps_; }
};
```

Override `invalidate(ControlFlowNode& changed)` and return `true` if the analysis can update itself after a local change instead of being recomputed.

Continue with [Passes](./passes.md) to see how docc changes the SDFG.
