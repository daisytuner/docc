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
| `InLocalStorage` | Copy read-only tile into contiguous local buffer before loop | Packing A/B panels in GEMM |
| `OutLocalStorage` | Create local tile buffer for write/read-write data, writeback after loop | Accumulator registers, output tiles |

Both transformations use `MemoryLayoutAnalysis` tile API to compute a bounding-box
over all accesses to the container within the target loop. The tile is delinearized
from the flat pointer access pattern (e.g. `A[i*K+j]` → shape `[extent_i, extent_j]`).

**Design principle:** All tile extents must resolve to integer constants via
`extents_approx()` (overapproximation). This means the performance engineer must
first establish a bounded tile region through `LoopTiling`, then apply packing at
the inner loop level where extents are determined by integer tile sizes. This
enforces the correct workflow: tiling defines the bounding box, packing materializes it.

### InLocalStorage

Creates a contiguous local copy of read-only data accessed in a loop:

```cpp
// Apply at inner loop level where tile extents are integer
InLocalStorage ils(i_loop, a_access);  // Pack tile of A
```

- Container must be **read-only** within the loop (no writes)
- Container must be `Pointer` or `Array` type (not `Scalar`)
- Copy-in loops are inserted before the target loop
- Body accesses are rewritten to use the local buffer with base-subtracted indices
- For `Pointer` types: re-linearizes copy-in/copy-out using layout strides

### OutLocalStorage

Creates a local tile buffer for write or read-write data:

```cpp
// Apply at inner loop level where tile extents are integer
OutLocalStorage ols(i_loop, c_access);  // Accumulate into local tile
```

- Container must have **writes** within the loop
- **Read-write** (accumulator): init loop copies tile before, writeback after
- **Write-only**: no init loop, only writeback after
- **Scalar** containers: creates a scalar local (no loops needed)
- For `Pointer` types: re-linearizes copy-in/copy-out using layout strides

## Example: BLIS-style GEMM

```cpp
// Phase 1: Loop restructuring (7 transformations)
// i→j→k  →  i_tile→k_tile→j_tile→i→k→j

// Phase 2: Packing (apply at inner loop levels where tile extents are integer)
InLocalStorage(i_loop, a_access);  // Pack A panel (MC x KC tile)
InLocalStorage(k_loop, b_access);  // Pack B panel (KC x NC tile)

// Phase 3: Register blocking (6 transformations)
LoopTiling(i, MR);            // Micro-tile i
LoopTiling(j, NR);            // Micro-tile j
// ... interchanges ...
OutLocalStorage(i_micro, c_access); // Register tile for C (MR x NR)
```

See `tests/transformations/optimizations/blocking_test.cpp` for complete examples.
