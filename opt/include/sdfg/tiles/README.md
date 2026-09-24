# Tile Algebra: Reasoning About Memory Levels

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

## The Layout

A *layout* is the function mapping tile elements to their memory locations.
Represented as `(shape, strides, offset)`, it denotes an affine map. The tiles module
uses `math::tensor::TensorLayout` as the single source of truth for this type
(`tiles::Layout` is an alias); a coordinate is addressed via `resolve_element`.

**Definition (Layout).** A layout of rank $d$ is a triple $L = (\mathbf{s}, \mathbf{t}, o)$ with a *shape* $\mathbf{s} \in \mathbb{N}^d$, a *stride* $\mathbf{t} \in \mathbb{Z}^d$, and an *offset* $o \in \mathbb{Z}$. It denotes the affine map

$$ L(x_0, \dots, x_{d-1}) \;=\; o + \sum_{k=0}^{d-1} t_k\, x_k, \qquad 0 \le x_k < s_k, $$

under the *colexicographic* convention: dim $0$ varies fastest.

> **Intuition.** $\mathbf{s}$ is *how many* elements lie along each dim, $\mathbf{t}$ is *how far apart* they sit in memory, and $o$ is *where the tile starts*. Consider the $32\times 32$ tile above. Row-major addressing places element $(i, j)$ at $1024 \cdot i + j$, so the tile's top-left corner $(i_{\text{outer}}, j_{\text{outer}})$ lies at offset $o = 1024 \cdot i_{\text{outer}} + j_{\text{outer}}$. Within the tile, dim $0$ (the column $j$) steps by $1$ and dim $1$ (the row $i$) by $1024$ — one full row — giving stride $(1, 1024)$.

`TensorLayout::total_elements()` is the coordinate count $\prod_k s_k$ and `resolve_element` evaluates the map above; `MultiDim`, `Linearized`, and `Transposed` buffers are affine layouts, while `Padded` and `Swizzle` add a shared-memory bank-conflict placement on top.

## Schedules and Memory Levels

The layout fixes the tile's *geometry*: which elements it covers and where they reside in memory.
It does not determine *which memory level the tile should be staged in*; that is schedule-dependent.
If a loop processes tiles independently across threads, each tile may reside privately in registers.
If the threads process one tile *cooperatively* (a shared operand or a reduction), the tile must reside in memory visible to every cooperating thread.

Geometry alone is therefore insufficient: **geometry and schedule together determine the minimum memory level** a tile may occupy -- the most local (fastest) space still visible to every thread that shares it.
The level is derived by classifying the enclosing loops of the tile.

- **Cooperation levels** $\mathrm{Device} \sqsupset \mathrm{Group} \sqsupset \mathrm{Subgroup}$. Target-neutral tiers (OpenCL/SYCL vocabulary) for the scope of work sharing: a GPU grid / CPU threads cooperate at **Device**, a GPU thread block / Tenstorrent core-workers at **Group**, a GPU warp / wavefront at **Subgroup**. Each target maps its schedule onto these tiers via `AxisSchedule::classify`.
- **Memory spaces** $\mathrm{Global} \sqsupset \mathrm{Shared} \sqsupset \mathrm{Register}$ — how widely a location is visible. Each target maps levels to spaces via `TileTarget::space`: a scratchpad target (GPU) sends $\mathrm{Device}\mapsto\mathrm{Global}$, $\mathrm{Group}\mapsto\mathrm{Shared}$, $\mathrm{Subgroup}\mapsto\mathrm{Register}$ (a subgroup cooperates through register shuffles, needing no buffer), while a flat host (CPU) target sends every level to $\mathrm{Global}$, having no scratchpad. `classify` stamps each axis with its target's space and a derived `has_scratchpad()` capability, so the algebra never names a target.

**Definition (axis role).** For an enclosing loop with index variable $x$, its role in staging the tile with base address $b$ is

$$ \operatorname{role}(x) = \begin{cases} \textbf{Cooperative} & x \notin \operatorname{vars}(b), \\ \textbf{Private} & x \in \operatorname{vars}(b). \end{cases} $$

> **Intuition.** The test is whether the thread index appears in the address. If it does, distinct threads access distinct data, so each holds a **private** copy. If it does not, all threads of the axis access the *same* data and must stage it **cooperatively**.

**Definition (required space).** A cooperatively-staged tile must live in a space visible to *every* thread that cooperates on it. So the tile's required space is the coarsest space over its cooperative axes, and thread-private (registers) when there are none:

$$ \operatorname{space}(T) \;=\; \bigsqcup \{\, \texttt{space}(\operatorname{level}(a)) : a \in \operatorname{axes}(T),\ a\ \text{cooperative} \,\}, $$

with $\bigsqcup$ the coarsest ($\mathrm{Global} > \mathrm{Shared} > \mathrm{Register}$) and the empty join $= \mathrm{Register}$. `Tile::required_space()` computes exactly this join. Concretely: cooperation across groups (device) $\Rightarrow$ Global; within a group $\Rightarrow$ Shared; within a subgroup only $\Rightarrow$ registers + shuffle; none $\Rightarrow$ a private register/stack block.

## LocalStorage: A Transformation to Stage Tiles into Memory Levels

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

## GPU Reduction Buffers

`tiles::ReductionBufferAnalysis` owns reduction layout and allocation inference.
Query it through `AnalysisManager::get<tiles::ReductionBufferAnalysis>()`:

- `buffer(reduce, container)` reports the original accumulator index, compact layout,
    element type, private bytes, shared bytes, and shared-buffer owner.
- `require(reduce, container)` requires an exact layout and throws with a diagnostic
    for unsupported or inconsistent materialization. Code generation uses this query;
    it has no fallback to padded linear spans.
- `kernel(root)` totals reduction shared-memory allocations below a kernel root,
    counting each shared owner once. It does not include unrelated staging buffers
    or compare against a device budget.
- `estimate(proposal)` evaluates an interchange, schedule change, or detached SDFG
    using fresh analyses without changing the original graph. Callers must not pass
    the live SDFG as the proposal.

Results distinguish `Exact`, `ConservativeBound`, and `Unsupported`. A conservative
bound can supply allocation costs but has no materializable layout. Unknown costs
are absent optionals, never zero. Products and kernel totals use checked arithmetic.
Compact indexing currently requires scalar or one-dimensional accumulator accesses
with consistent indices and constant positive affine output axes. Guarded tile
counts are accepted when assumptions prove them exact; scatter reductions are rejected.

`passes::ReductionSharedMemoryDelinearization` creates real private (`CPU_Stack`)
and shared (`NV_Shared`) arrays and rewrites accumulator memlets to compact indices.
`ReductionInfo::container` retains the original writeback target; its serialized
`original_index` preserves the original address relation. User and dependency analyses
therefore see the writeback even though body accesses now name partial buffers.
The schedule stores only strategy and buffer references (`reduction_private.<name>`
and `reduction_shared.<name>`), not inferred sizes. Ordinary array declarations are
validated against fresh inference after materialization or cloning.

The Python compile/source-emission paths and LLVM compilation invoke the pass
immediately before instrumentation planning and code generation (after graph splitting
in LLVM). Scheduling does not materialize buffers, so compilation also accepts
already-scheduled graphs that bypass RPC or heuristic scheduling. Low-level generator
and dispatcher callers must prepare their SDFGs explicitly; dispatchers do not
materialize reduction buffers. Repeated calls validate existing materialization without
changing it. The reduction dispatcher remains the single owner of identity
initialization, synchronization, combine and publication; there is no language-extension
access remapping. Packed memlets use the existing GPU thread-index builtins directly,
without reduction-specific aliases.

Interchange, tiling, GPU offload and local-storage transformations preview affected
footprints before changing the live graph and invalidate analyses after applying.
Perform these optimizations before packing: incompatible changes after materialization
are rejected, not automatically unpacked and repacked. Analysis results and references
must not be retained across invalidation. Hardware-budget selection and automatic
Shared-to-Global fallback are separate policy work, not part of this lowering.

## The Tile API: Build Your Own Transformations

`LocalStorage` is one transformation built on the tile algebra; the same API supports others (double buffering, asynchronous pipelines, custom packings). All types below are pure values with no SDFG state, so a movement plan is assembled and only the final `TileCopyNode` emission modifies the graph.

- **`Tile`** — the output of `TileAnalysis`. It pairs the tile's *source layout* (`TileInfo::source_layout()`, the global gather geometry) with the classified schedule axes and reports `required_space()` (the minimum level from the previous section). It is the input to a transformation: one staged region, described geometrically and schedule-classified.

- **`PackedBuffer`** — the *realized layout*: the concrete buffer allocated at the target level. Where `Tile.source` describes the location of data in the *global* array, `PackedBuffer` is the destination layout in materialized form. It provides the buffer **type** via `axes()` (one extent per nested-array level) and the element **address** via `subset(slot, tile)` (one index per level), both consistent with its scalar-offset `layout()`. Its `kind` selects the layout of each per-thread slot's block:

  - **`MultiDim`** — a dense row-major nested array `[slot][tile…]`. The default: representing each dimension as a separate array level allows the compiler to recover per-axis strides and vectorize. A pure (affine) layout.
  - **`Linearized`** — the same data as a single flat axis. Required by the CDNA asynchronous `global_load_lds` DMA, which writes lane-contiguous from a wave-uniform base and therefore requires a flat, lane-ordered buffer. A pure (affine) layout.
  - **`Padded`** — inflates the per-slot inner stride to a value coprime with 32, so that a warp's per-slot accesses map to *distinct* shared-memory banks (a bank is `address mod 32`; collisions serialize). The layout remains affine; only the padding amount is a hardware heuristic rather than a consequence of the geometry.
  - **`Swizzle`** — XORs the inner index with the slot index, distributing banks without unused columns. Since XOR is non-linear, this placement lies *outside* the affine algebra: a `Swizzle` functor composed with a `Layout` (`ComposedLayout = swizzle ∘ layout`), applied identically to writes and reads and therefore a pure relabelling of storage locations.
  - **`Transposed`** — dense, but the tile axes are stored *column-major* (reversed), so a logical `[M][N]` tile lives physically as `[N][M]`. A consumer that reads `buf[m][n]` then walks memory in the orthogonal direction (coalesced/​bank-friendly reads for the transposed operand) without a separate transpose pass. Still affine — only the tile strides are permuted — and, like the swizzle, applied identically to the copy and the read.

  `MultiDim`, `Linearized`, and `Transposed` are pure affine layouts; `Padded` and `Swizzle` are bank-conflict-avoidance placements for shared memory.

- **`TiledCopy`** — the *movement plan*: two geometry layouts `src` (global) and `dst` (buffer), a *copy atom* (`ScalarSync`, `VectorSync`, or `CpAsync`), and an optional `dst_swizzle` (the XOR placement carried on the buffer offset). `src` maps a tile coordinate to the global element and `dst` to the local-buffer slot; the atom fixes the per-lane transfer.

- **`TileCopyNode`** — the plan materialized as *one library node*, the only step that modifies the graph. A transformation assembles a `TiledCopy` and emits a single `TileCopyNode`; its backend dispatcher lowers the whole cooperative copy (the thread-strided staging loop, the atom's transfer, and the boundary guard) at codegen time. Two rules keep the node self-contained:

  - **Bare-pointer memlets.** The `{_dst, _src}` inputs are empty-subset computational memlets — plain base pointers, *not* a dereference. All addressing lives in the plan (`src`/`dst` layouts, offsets, `dst_swizzle`), so `symbols()`/`replace()` rewrite the plan directly and a pass (double-buffering, pipelining) can retarget the copy by editing the plan without touching surrounding control flow.
  - **Cooperation modes.** `coop_axes` selects how the block splits the tile: empty = *whole-block* (every thread strides the flat tile — a slot-free cooperative tile or the lane-contiguous DMA); a subset = *per-thread-slot* (each thread stages its own slot over the listed spatial axes, so a mixed per-thread + cooperative tile composes with the ragged per-thread guard). A `Swizzle` buffer rides on `dst_swizzle`, applied identically to the copy write and the compute read.

  `SoftwarePipelining` double-buffers a node by biasing `plan.dst.offset` per stage and flipping the atom to `CpAsync`; `TileVectorizer` widens it by mutating the atom/bytes — both derive legality from the plan, never from a sibling subgraph.


## Adding a Target: the `TileTarget` Interface

Everything above is target-neutral: the levels, spaces, `Layout`, `Tile`, `PackedBuffer`, and `TiledCopy` never name a backend. A backend (CUDA, ROCm, OpenMP, Tenstorrent, or an externally linked one) makes its schedules and memory legible to that neutral core by implementing a single interface, `TileTarget`, and registering it — **no edit to the tile core is required**. A target answers five questions:

- **`classify(ScheduleType) → optional<AxisSchedule>`** — how one of its loop schedules cooperates: the cooperation `Level`, its backing `Space`, spatial dimension (X/Y/Z), parallel size, and sync need. Returns `nullopt` for a schedule that does not shape storage (a sequential loop). This is the *only* decoder from a raw schedule value into the neutral axis vocabulary.
- **`space(Level) → Space`** — which memory tier backs cooperation at each level: a GPU maps $\mathrm{Group}\mapsto\mathrm{Shared}$, $\mathrm{Subgroup}\mapsto\mathrm{Register}$; a flat CPU maps every level to $\mathrm{Global}$; an exotic scratchpad target maps whichever levels it materializes on-chip. The derived `has_scratchpad()` follows from this map, letting the algebra tell a device-wide axis that sits *atop* a scratchpad (a GPU grid) from a flat host axis — without naming the target.
- **`supports_cooperative_staging(ScheduleType) → bool`** — whether a schedule can host a group-cooperative staging copy driven by its own threads (a genuine offload schedule), versus a fused whole-kernel schedule that cooperates at group level but cannot carry a separate copy map. This is the one distinction `classify` alone cannot make, since both land at `Level::Group`.
- **`storage_type(Space) → StorageType`** — the concrete buffer that realizes an abstract tier: $\mathrm{Shared}\mapsto$ `NV_Shared` / `TT_L1`, $\mathrm{Global}\mapsto$ `NV_Global`, $\mathrm{Register}\mapsto$ a thread-private stack/register block.
- **`lane_width() → unsigned`** — the SIMD lane / subgroup width (32 on NVIDIA, 64 on CDNA, 1 where there is no SIMD cooperation), used only to keep a subgroup's cooperative stores bank-conflict-free.

A target registers its `TileTarget` in its `register_*_plugin`, under **each schedule value it owns** (CUDA registers one instance under both `"CUDA"` and `"CUDA_Offload"`), via the `TileTargetRegistry` singleton. `AxisSchedule::classify` resolves the owner by schedule value and delegates to it; an unregistered schedule falls back to a neutral, no-scratchpad rule. Because this interface is the sole seam, the hard-coded `"CUDA_Offload"` / `NV_Shared` / warp-size checks that once scattered through the tile core now live behind it, and a new backend plugs in by implementing five methods and registering — with no change to `opt`.
