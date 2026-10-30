---
title: Passes
sidebar_position: 4
description: How docc rewrites whole SDFGs - passes, pipelines, library node expansion, and scheduling
---

A **pass** rewrites an entire Structured SDFG: it walks the SDFG, finds every place where a rewrite applies, and applies it. Passes simplify the SDFG, expand high-level operations, and finally schedule loops for a target. The compilation pipelines of all frontends are sequences of passes.

Compared to a [transformation](./transformations.md), which changes one place that you choose, a pass decides itself where to apply. Many passes are built from transformations and analyses.

## Example: from loop to parallel code

The Python bindings group the passes of the compilation pipeline into a few steps. Applied to WAXPBY from [Structured SDFGs](./structured-sdfgs.md#example-waxpby):

```python
from waxpby_sdfg import build_waxpby

sdfg = build_waxpby().move()
print(sdfg.root.at(0))   # <For indvar='i' ...>
sdfg.simplify()
print(sdfg.root.at(0))   # <Map indvar='i' schedule='SEQUENTIAL' ...>
sdfg.normalize()
sdfg.schedule("openmp", "server", False)
print(sdfg.root.at(0))   # <Map indvar='i' schedule='CPU_PARALLEL' ...>
```

| Step          | What happens                                                                                                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `expand()`    | Expands library nodes into loops and tasklets where the target has no better implementation                     |
| `simplify()`  | Runs the simplification pipelines (dataflow, symbolic, dead code, constants). Its last pass, `AutoParallelization`, turns every `For` whose iterations are independent into a `Map`. |
| `normalize()` | Brings loops into a canonical form and fuses loops and blocks where possible                                    |
| `schedule()`  | Runs the loop schedulers of the target, which assign a schedule to each `Map`, here OpenMP                      |

These are the stages that `DOCC_DEBUG=dump` writes for the Python and PyTorch frontends, see [Debug Options](../../python/debugging.md#look-at-the-intermediate-steps). The C/C++ frontend runs the same kinds of passes inside the LLVM plugin.

## The `Pass` interface

A pass derives from `passes::Pass` and implements `run_pass`, which returns whether it changed the SDFG:

```cpp
class Pass {
public:
    virtual std::string name() = 0;
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);
    virtual void invalidates(analysis::AnalysisManager& analysis_manager, bool applied);
    virtual std::vector<OptionSpec> options();   // options the pass can be configured with
};
```

`run(builder, analysis_manager)` calls `run_pass` and then `invalidates`, which by default discards all cached [analyses](./analyses.md) if the pass changed something. In Python, individual passes are available as classes, for example `SymbolPropagation().run(builder, analysis_manager)`.

## Pipelines

A `Pipeline` is a pass that runs a list of passes. It runs each pass repeatedly until it no longer changes the SDFG, and the whole list repeatedly until none of its passes changes anything. Predefined pipelines in `sdfg/include/sdfg/passes/pipeline.h`:

| Pipeline                     | Purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| `dataflow_simplification`    | Fuse tasklets and blocks, propagate references, remove trivial arrays   |
| `symbolic_simplification`    | Propagate and simplify symbols and their expressions                    |
| `dead_code_elimination`      | Remove unreachable control flow and unused data                         |
| `constant_elimination`       | Propagate constants                                                     |
| `controlflow_simplification` | Simplify conditions and loop structures                                 |
| `memlet_combine`             | Simplify and linearize memlet subsets                                   |
| `memory`                     | Manage allocations                                                      |

## Kinds of passes

| Directory in `sdfg/include/sdfg/passes/` | Examples                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ |
| `dataflow/`                              | `TaskletFusion`, `ConstantPropagation`, `DeadDataElimination`, `ApproximateMath`             |
| `structured_control_flow/`               | `BlockFusion`, `SequenceFusion`, `WhileToForConversion`, `DeadCFGElimination`                |
| `symbolic/`                              | `SymbolPropagation`, `SymbolPromotion`, `SymbolEvolution`                                    |
| `memory/`                                | `AllocationManagement`, `TensorAllocationSizeInference`                                      |
| `expansion/`                             | `LibraryNodeExpansionPass`: library nodes such as `gemm` or `einsum` become loops            |
| `scheduler/`                             | `LoopScheduler`, the interface for target schedulers                                         |

Passes that need `opt/`, for example `AutoParallelization`, loop fusion, normalization, and the target schedulers, live in `opt/include/sdfg/passes/` and `opt/include/sdfg/parallelization/passes/`.

## Scheduling

Scheduling maps loops onto the target. The `LoopSchedulingPass` walks the loop tree from the outermost loops inward and asks the target's `LoopScheduler` for each loop:

1. `find`: should this loop be scheduled (`APPLY`), skipped (`NEXT`), or should its children be considered (`CHILDREN`)?
2. `can_apply_schedule` and `apply_schedule`: apply the schedule, usually with [transformations](./transformations.md).

The OpenMP scheduler, for example, applies `OMPTransform` to the outermost `Map` of a loop nest. The CUDA and ROCm schedulers offload loop nests to the GPU and map them to thread blocks and threads. Each target registers its schedulers through the [plugin interface](../plugins.md).

## Pass options

Passes can declare options (`Pass::options()`), for example a work budget for the symbolic analysis. They are registered globally, listed in Python with `docc.sdfg.registered_options()`, and set through the `pass_options` option of the frontends, see [Python Usage](../../python/usage.md#pass-options).

## Write a pass

A pass that removes empty blocks from the root sequence:

```cpp
#include "sdfg/passes/pass.h"

class RemoveEmptyBlocks : public sdfg::passes::Pass {
public:
    std::string name() override { return "RemoveEmptyBlocks"; }

    bool run_pass(sdfg::builder::StructuredSDFGBuilder& builder,
                  sdfg::analysis::AnalysisManager& analysis_manager) override {
        bool applied = false;
        auto& root = builder.subject().root();
        for (size_t i = 0; i < root.size();) {
            auto* block = dynamic_cast<sdfg::structured_control_flow::Block*>(&root.at(i));
            if (block && block->dataflow().nodes().begin() == block->dataflow().nodes().end()) {
                builder.remove_child(root, i);
                applied = true;
            } else {
                i++;
            }
        }
        return applied;
    }
};
```

All changes go through the builder. For passes that only need to visit certain node types, derive from `VisitorPass<T>` with a visitor class `T`. Existing passes in `sdfg/src/passes/` are good starting points.

Continue with [Transformations](./transformations.md) to change one specific loop.
