# SDFG Optimizations

This module provides transformations for optimizing Structured Data Flow Graphs (SDFGs).

## Transformations API

All transformations inherit from `Transformation` and implement:

```cpp
class MyTransformation : public Transformation {
public:
    MyTransformation(/* target loop/node */, /* parameters */);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& am) override;

    void apply(builder::StructuredSDFGBuilder& builder,
               analysis::AnalysisManager& am) override;
};
```

### Usage Pattern

```cpp
// 1. Create transformation with target and parameters
transformations::LoopTiling tiling(loop, tile_size);

// 2. Check if applicable
if (tiling.can_be_applied(builder, am)) {
    // 3. Apply transformation
    tiling.apply(builder, am);
    am.invalidate_all();  // Invalidate cached analyses
}

// Or use the Recorder for history tracking:
recorder.apply<transformations::LoopTiling>(builder, am, false, loop, tile_size);
```

## Loop Transformations

| Transformation | Description | Parameters |
|----------------|-------------|------------|
| `LoopTiling` | Split loop into tile/point loops | `(loop, tile_size)` |
| `LoopInterchange` | Swap two nested loops | `(outer_loop, inner_loop)` |
| `LoopDistribute` | Split loop into multiple loops | `(loop)` |
| `LoopSkewing` | Apply skewing for wavefront parallelism | `(loop, skew_factor)` |
| `LoopRotate` | Rotate loop body iterations | `(loop)` |
| `LoopShift` | Shift loop iteration space | `(loop, shift_amount)` |
| `CollapseToDepth` | Collapse nested loops to target depth | `(loop, depth)` |

## Packing / Local Storage Transformations

| Transformation | Description | Use Case |
|----------------|-------------|----------|
| `InLocalStorage` | Pack read-only data into local buffer | A/B matrices in GEMM |
| `OutLocalStorage` | Scalar accumulator for loop-invariant writes | Reduction variables |
| `AccumulatorTile` | Multi-element tile for read-write patterns | C tile in micro-kernel |

### InLocalStorage
Creates a local copy of data accessed in a loop, with optional copy loops:
```cpp
InLocalStorage ils(target_loop, "A");  // Pack array A
```

### OutLocalStorage
Creates a scalar or small buffer for accumulation when the output index is constant w.r.t. the target loop:
```cpp
OutLocalStorage ols(inner_loop, "y");  // Scalar accumulator for y[i]
```

### AccumulatorTile
Creates a tile buffer for read-write patterns where output depends on inner loop indices:
```cpp
AccumulatorTile acc(micro_tile_loop, "C");  // MR×NR register tile
```

## Example: BLIS-style GEMM

```cpp
// Phase 1: Loop restructuring (7 transformations)
// i→j→k  →  i_tile→k_tile→j_tile→i→k→j

// Phase 2: Packing (2 transformations)
InLocalStorage(k_tile, "A");  // Pack A panel
InLocalStorage(j_tile, "B");  // Pack B panel

// Phase 3: Register blocking (6 transformations)
LoopTiling(i, MR);            // Micro-tile i
LoopTiling(j, NR);            // Micro-tile j
// ... interchanges ...
AccumulatorTile(i_micro, "C"); // Register tile for C
```

See `tests/transformations/optimizations/blocking_test.cpp` for complete examples.
