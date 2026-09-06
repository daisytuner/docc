# sdfg-opt: Transformations for Optimization of SDFGs

This module provides transformations for the optimization of SDFGs.
This comprises four core components:

1. **Transformations:** An API to modify SDFGs.
2. **Loop Scheduling:** An API to for loop scheduling using symbolic analysis and the polyhedral framework.
3. **Tile Algebra:** An algebra of tiles implemented as an analysis to implement packing and pipelining transformations.
4. **Default Targets:** The OpenMP, CUDA and ROCm targets for parallel schedules on SDFGs.

## Transformations: An API to Modify SDFGs

Transformations are a structured, unified way to modify SDFGs under correctness constraints.
A transformation is a class inheriting from the abstract `Transformation`.

```cpp
class MyTransformation : public Transformation {
public:
    MyTransformation(
        /* subgraph */,
        /* parameters */
    );

    bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& am) override;

    void apply(builder::StructuredSDFGBuilder& builder,
               analysis::AnalysisManager& am) override;

    void to_json(nlohmann::json& j) const override;

    static MyTransformation from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

};
```

```cpp
// Explicit: Create transformation with target and parameters
transformations::LoopTiling tiling(loop, tile_size);
if (tiling.can_be_applied(builder, am)) {
    tiling.apply(builder, am);
}

// Recorder:
recorder.apply<transformations::LoopTiling>(builder, am, false, loop, tile_size);
```

## Loop Scheduling

TODO

## Tile Algebra: Reasoning About Memory Levels

The dominant bottleneck of modern processors is usually memory, not compute.
Moving data through the memory hierarchy as fast as possible -- on a CPU, from DRAM (global memory) up through several cache levels into the registers the ALU operates on -- is therefore central to performance.
On a CPU this movement is largely automatic: the compiler and hardware prefetchers steer the caches, leaving little to control explicitly.
On GPUs and TPUs it is frequently explicit: data must be moved from global memory into shared memory / SRAM by hand.

This module provides a mathematical model for data in the memory hierarchy and for building transformations on it.
The core unit is the **tile**: a compile-time-bounded region of memory.
Consider copying a 1024x1024 matrix from `A` to `B`, both in DRAM.
The naive nest below already defines tiles -- the full matrices `A`, `B`, and, per iteration, the rows `A_i`, `B_i` -- but the compiler and prefetcher already move these optimally, so explicit staging yields no benefit:

```c++
void copy(float* A, float* B) {
    // tiles: matrices A, B
    for (int i = 0; i < 1024; i++) {
        // tiles: rows A_i, B_i
        for (int j = 0; j < 1024; j++) {
            B[i * 1024 + j] = A[i * 1024 + j];
        }
    }
}
```

The loop transformations introduced earlier produce different tiles without changing program semantics:

```c++
void copy(float* A, float* B) {
    // tiles: matrices A, B
    for (int i_outer = 0; i_outer < 1024; i_outer += 32) {
        for (int j_outer = 0; j_outer < 1024; j_outer += 32) {
            // tile: a 32x32 2D subset
            for (int i = i_outer; i < i_outer + 32; i++) {
                for (int j = j_outer; j < j_outer + 32; j++) {
                    B[i * 1024 + j] = A[i * 1024 + j];
                }
            }
        }
    }
}
```

Strip-mining and loop interchange produce a 32x32 tile, which provides an explicit handle on data movement.
The staging pattern is then almost always the same:
- **Allocate** a tile-sized buffer on the lower level (shared memory, stack, SRAM).
- **Copy in** the data from the higher level before the loop defining the tile.
- **Copy out** the data back to the higher level after the loop defining the tile.

This is the core pattern of tiling and local storage.
The remaining complexity consists of variants of it:
- cooperative vs. thread-private loads,
- cooperative stores (reductions, atomics),
- double buffering / asynchronous loads and stores (staging the next tile while processing the current one).

Identifying which accesses form compile-time-bounded regions inside a loop is implemented in `MemoryLayoutAnalysis`.
Classifying those regions into schedule- and storage-level-aware tiles is implemented in `TileAnalysis`.

### The Layout Algebra

A *layout* is the function mapping tile elements to their memory locations.
Represented as `(shape, stride, offset)`, layouts compose and normalize under a small set of operations, and their legality — injectivity and exact cover — is checkable symbolically.

**Definition (Layout).** A layout of rank $d$ is a triple $L = (\mathbf{s}, \mathbf{t}, o)$ with a *shape* $\mathbf{s} \in \mathbb{N}^d$, a *stride* $\mathbf{t} \in \mathbb{Z}^d$, and an *offset* $o \in \mathbb{Z}$. It denotes the affine map

$$ L(x_0, \dots, x_{d-1}) \;=\; o + \sum_{k=0}^{d-1} t_k\, x_k, \qquad 0 \le x_k < s_k, $$

under the *colexicographic* convention: dim $0$ varies fastest.

> **Intuition.** $\mathbf{s}$ is *how many* elements lie along each dim, $\mathbf{t}$ is *how far apart* they sit in memory, and $o$ is *where the tile starts*. Consider the $32\times 32$ tile above. Row-major addressing places element $(i, j)$ at $1024 \cdot i + j$, so the tile's top-left corner $(i_{\text{outer}}, j_{\text{outer}})$ lies at offset $o = 1024 \cdot i_{\text{outer}} + j_{\text{outer}}$. Within the tile, dim $0$ (the column $j$) steps by $1$ and dim $1$ (the row $i$) by $1024$ — one full row — giving stride $(1, 1024)$.

Two derived quantities matter:

- the **size** $\lvert L \rvert = \prod_k s_k$ — the number of coordinates (its *domain*);
- the **cosize** $\operatorname{co} L = o + \sum_k (s_k - 1)\, t_k + 1$ — the extent of memory it touches (the smallest buffer that holds its image).

Always $\lvert L \rvert \le \operatorname{co} L$, with equality exactly when $L$ is a dense, gap-free block.

**Definition (injective, bijective).** $L$ is *injective* if $L(x) = L(y) \Rightarrow x = y$, and *bijective* if additionally its image is exactly $[o,\, o + \lvert L \rvert)$ — a dense permutation.

> **Intuition.** Injectivity is the *no-double-write* property: a destination layout must not map two tile coordinates to the same slot. Bijectivity is *exact cover*: a `(thread, value)` partition of a tile must map onto every element exactly once. `is_injective` / `is_bijective` decide these conservatively, returning `false` whenever the symbolic side-condition cannot be established.

**The operations.** Two total operators combine and normalize layouts:

| Operation | Notation | Meaning | Intuition |
|---|---|---|---|
| Concatenation | `A ++ B` | append `B`'s dims at absolute strides | place two layouts side by side without rescaling |
| Coalesce | `coalesce(A)` | merge contiguous dims, drop size-1 dims | canonical normal form; $\operatorname{coalesce}(A)(i) = A(i)$ |

Coalesce is idempotent and function-preserving, giving every layout a canonical representative.

> **Remark.** Concatenation is how a `(thread, value)` partition is assembled: `partition = A ++ B` places the value layout beside the thread layout, and `is_bijective` then checks that the pair covers the tile exactly. In the SDFG setting the loop nest itself expresses the tiling hierarchy (via strip-mining and loop interchange), so the layout only needs to describe a single flat region rather than carry a nested structure of its own.

### Schedules and Memory Levels

The layout fixes the tile's *geometry*: which elements it covers and where they reside in memory.
It does not determine *which memory level the tile should be staged in*; that is schedule-dependent.
If a loop processes tiles independently across threads, each tile may reside privately in registers.
If the threads process one tile *cooperatively* (a shared operand or a reduction), the tile must reside in memory visible to every cooperating thread.

Geometry alone is therefore insufficient: **geometry and schedule together determine the minimum memory level** a tile may occupy -- the most local (fastest) space still visible to every thread that shares it.
The level is derived by classifying the enclosing loops of the tile.

- **Cooperation levels** $\mathrm{Device} \sqsupset \mathrm{Group} \sqsupset \mathrm{Subgroup}$. Target-neutral tiers (OpenCL/SYCL vocabulary) for the scope of work sharing: a GPU grid / CPU threads cooperate at **Device**, a GPU thread block / Tenstorrent core-workers at **Group**, a GPU warp / wavefront at **Subgroup**. Each target maps its schedule onto these tiers via `AxisSchedule::classify`.
- **Memory spaces** $\mathrm{Global} \sqsupset \mathrm{Shared} \sqsupset \mathrm{Register}$ — how widely a location is visible. Each target maps levels to spaces via `TileTarget::space`: a scratchpad target (GPU) sends $\mathrm{Device}\mapsto\mathrm{Global}$, $\mathrm{Group}\mapsto\mathrm{Shared}$, $\mathrm{Subgroup}\mapsto\mathrm{Register}$ (the canonical `default_space` hierarchy; a subgroup cooperates through register shuffles, needing no buffer), while a flat host (CPU) target sends every level to $\mathrm{Global}$, having no scratchpad. `classify` stamps each axis with its target's space and a derived `has_scratchpad()` capability, so the algebra never names a target.

**Definition (axis role).** For an enclosing loop with index variable $x$, its role in staging the tile with base address $b$ is

$$ \operatorname{role}(x) = \begin{cases} \textbf{Cooperative} & x \notin \operatorname{vars}(b), \\ \textbf{Private} & x \in \operatorname{vars}(b). \end{cases} $$

> **Intuition.** The test is whether the thread index appears in the address. If it does, distinct threads access distinct data, so each holds a **private** copy. If it does not, all threads of the axis access the *same* data and must stage it **cooperatively**.

**Definition (required space).** A cooperatively-staged tile must live in a space visible to *every* thread that cooperates on it. So the tile's required space is the coarsest space over its cooperative axes, and thread-private (registers) when there are none:

$$ \operatorname{space}(T) \;=\; \bigsqcup \{\, \texttt{space}(\operatorname{level}(a)) : a \in \operatorname{axes}(T),\ a\ \text{cooperative} \,\}, $$

with $\bigsqcup$ the coarsest ($\mathrm{Global} > \mathrm{Shared} > \mathrm{Register}$) and the empty join $= \mathrm{Register}$. `Tile::required_space()` computes exactly this join. Concretely: cooperation across groups (device) $\Rightarrow$ Global; within a group $\Rightarrow$ Shared; within a subgroup only $\Rightarrow$ registers + shuffle; none $\Rightarrow$ a private register/stack block.

### LocalStorage: A Transformation to Stage Tiles into Memory Levels

`TileAnalysis` *derives* the minimum level; `LocalStorage` is the transformation that *enforces* it.
Given a tile, it allocates a buffer in the derived space, rewrites the loop body to access that buffer, and inserts the copy-in before and copy-out after the loop.

| Coarsest cooperative axis | Read tile | Written tile |
|---|---|---|
| none (private / sequential) | private register/stack block | private, copied back after the loop |
| Subgroup | registers + shuffle — no staged buffer | register reduction |
| Group | block-shared buffer, staged cooperatively + barrier | reduction (owned by the reduce dispatcher) |
| Device | grid-global buffer (each group stages its own) | reduction / atomic merge |

When the minimum level is achieved *without* a buffer — a subgroup sharing through register shuffles, or a cooperative write realized as a reduction/atomic — there is nothing to stage, and `LocalityPlan::required_space` declines; these cases are handled by the shuffle and reduce machinery.

> **Example.** In `out[i] += A[k]` with `i` mapped across a GPU block and `k` sequential, the tile of `A` has base $b = k$, independent of `i`. Since `i` does not appear in $b$, the `i` axis is **cooperative** at the **Group** level (a GPU thread block), so `A` is staged **once** into a **Shared** buffer, filled cooperatively by the block's threads. For the access `A[i*16 + k]`, the base $b = 16i$ depends on `i`: the axis is **Private**, and each thread stages its own row-tile, without sharing or a barrier.

### The Tile API: Build Your Own Transformations

`LocalStorage` is one transformation built on the tile algebra; the same API supports others (double buffering, asynchronous pipelines, custom packings). All types below are pure values with no SDFG state, so a movement plan is assembled and only the final `emit` step modifies the graph.

- **`Tile`** — the output of `TileAnalysis`. It pairs the tile's *source layout* (`TileInfo::source_layout()`, the global gather geometry) with the classified schedule axes and reports `required_space()` (the minimum level from the previous section). It is the input to a transformation: one staged region, described geometrically and schedule-classified.

- **`PackedBuffer`** — the *realized layout*: the concrete buffer allocated at the target level. Where `Tile.source` describes the location of data in the *global* array, `PackedBuffer` is the destination layout in materialized form. It provides the buffer **type** via `axes()` (one extent per nested-array level) and the element **address** via `subset(slot, tile)` (one index per level), both consistent with its scalar-offset `layout()`. Its `kind` selects the layout of each per-thread slot's block:

  - **`MultiDim`** — a dense row-major nested array `[slot][tile…]`. The default: representing each dimension as a separate array level allows the compiler to recover per-axis strides and vectorize. A pure (affine) layout.
  - **`Linearized`** — the same data as a single flat axis. Required by the CDNA asynchronous `global_load_lds` DMA, which writes lane-contiguous from a wave-uniform base and therefore requires a flat, lane-ordered buffer. A pure (affine) layout.
  - **`Padded`** — inflates the per-slot inner stride to a value coprime with 32, so that a warp's per-slot accesses map to *distinct* shared-memory banks (a bank is `address mod 32`; collisions serialize). The layout remains affine; only the padding amount is a hardware heuristic rather than a consequence of the geometry.
  - **`Swizzle`** — XORs the inner index with the slot index, distributing banks without unused columns. Since XOR is non-linear, this placement lies *outside* the affine algebra: a `Swizzle` functor composed with a `Layout` (`ComposedLayout = swizzle ∘ layout`), applied identically to writes and reads and therefore a pure relabelling of storage locations.

  `MultiDim` and `Linearized` are pure affine layouts (selected for correctness and the DMA constraint); `Padded` and `Swizzle` are bank-conflict-avoidance placements for shared memory.

- **`TiledCopy`** — the *movement plan*: four layouts `(src, dst, thread, value)` and a *copy atom* (`ScalarSync`, `VectorSync`, `CpAsync`, or the CDNA `LaneContiguousDMA`). `src` and `dst` are the global and buffer geometries; `(thread, value)` partition the tile across the lanes. Its `verify(elem_bytes)` establishes correctness before any device code is generated: a copy is valid iff

  1. source and destination span the same tile, $\lvert \text{src} \rvert = \lvert \text{dst} \rvert$;
  2. the partition `value ++ thread` is *bijective* onto the tile (exact cover);
  3. the destination is *injective* (no double-write); and
  4. the atom's width and contiguity constraints hold — including *lane-contiguity* (`dst ∘ thread` unit-stride) for the CDNA DMA.

  These conditions reduce two classes of GPU faults — a `cp.async` with an illegal transfer width and a lane-scrambled `global_load_lds` — to host-side unit tests.

- **`emit` / `emit_into`** — the only step that modifies the graph: it materializes a verified `TiledCopy` into the SDFG (the coverage map, the guarded element copy, and optional barriers). A transformation assembles the four objects above and passes the plan to `emit`.

### Adding a Target: the `TileTarget` Interface

Everything above is target-neutral: the levels, spaces, `Layout`, `Tile`, `PackedBuffer`, and `TiledCopy` never name a backend. A backend (CUDA, ROCm, OpenMP, Tenstorrent, or an externally linked one) makes its schedules and memory legible to that neutral core by implementing a single interface, `TileTarget`, and registering it — **no edit to the tile core is required**. A target answers five questions:

- **`classify(ScheduleType) → optional<AxisSchedule>`** — how one of its loop schedules cooperates: the cooperation `Level`, its backing `Space`, spatial dimension (X/Y/Z), parallel size, and sync need. Returns `nullopt` for a schedule that does not shape storage (a sequential loop). This is the *only* decoder from a raw schedule value into the neutral axis vocabulary.
- **`space(Level) → Space`** — which memory tier backs cooperation at each level: a GPU maps $\mathrm{Group}\mapsto\mathrm{Shared}$, $\mathrm{Subgroup}\mapsto\mathrm{Register}$; a flat CPU maps every level to $\mathrm{Global}$; an exotic scratchpad target maps whichever levels it materializes on-chip. The derived `has_scratchpad()` follows from this map, letting the algebra tell a device-wide axis that sits *atop* a scratchpad (a GPU grid) from a flat host axis — without naming the target.
- **`supports_cooperative_staging(ScheduleType) → bool`** — whether a schedule can host a group-cooperative staging copy driven by its own threads (a genuine offload schedule), versus a fused whole-kernel schedule that cooperates at group level but cannot carry a separate copy map. This is the one distinction `classify` alone cannot make, since both land at `Level::Group`.
- **`storage_type(Space) → StorageType`** — the concrete buffer that realizes an abstract tier: $\mathrm{Shared}\mapsto$ `NV_Shared` / `TT_L1`, $\mathrm{Global}\mapsto$ `NV_Global`, $\mathrm{Register}\mapsto$ a thread-private stack/register block.
- **`lane_width() → unsigned`** — the SIMD lane / subgroup width (32 on NVIDIA, 64 on CDNA, 1 where there is no SIMD cooperation), used only to keep a subgroup's cooperative stores bank-conflict-free.

A target registers its `TileTarget` in its `register_*_plugin`, under **each schedule value it owns** (CUDA registers one instance under both `"CUDA"` and `"CUDA_Offload"`), via the `TileTargetRegistry` singleton. `AxisSchedule::classify` resolves the owner by schedule value and delegates to it; an unregistered schedule falls back to a neutral, no-scratchpad rule. Because this interface is the sole seam, the hard-coded `"CUDA_Offload"` / `NV_Shared` / warp-size checks that once scattered through the tile core now live behind it, and a new backend plugs in by implementing five methods and registering — with no change to `opt`.
