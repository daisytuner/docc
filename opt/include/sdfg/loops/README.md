# Loops: Reshaping Iteration Spaces Without Reordering Them

A loop header is a compact description of a set of iterations — where to start, how far to step, when to stop — and the same set of iterations can be described in many ways.
These three loops all run the body for the values `10, 12, 14, 16, 18`, in that order:

```c++
for (int i = 10; i != 20; i += 2)  body(i);
for (int i = 0;  i < 5;   i++)     body(2 * i + 10);
for (int t = 0;  t < 5;   t += 2)  for (int i = t; i < t + 2 && i < 5; i++)  body(2 * i + 10);
```

The first is how a frontend might hand the loop to us; the second is the canonical form analyses prefer; the third is the same loop strip-mined into tiles of two.
Each rewrite changes the *description* of the iteration space, but every instance of `body` still runs exactly once and in the same order.
This module collects those rewrites.

## Order-Preserving Transformations

**Definition (order-preserving transformation).** A transformation is *order-preserving* if there is a bijection $\varphi$ between the statement instances before and after it, such that every instance keeps its memory accesses (after substituting the new induction variables) and

$$
a \prec b \iff \varphi(a) \prec \varphi(b),
$$

where $\prec$ is the execution order of the program.

By the Fundamental Theorem of Dependence (see the [reordering README](../reordering/README.md)), a transformation that never changes the relative order of two instances preserves every dependence.
**Order-preserving transformations are therefore legal without any dependence analysis.**
Their `can_be_applied` checks ask a different question: *can the new iteration space be expressed?* — is the bound affine, the stride a known constant, the trip count constant, the nest perfect?

> **Intuition.** These transformations rename iterations; they never shuffle them. A rename cannot make a read see a different write, so the only way a rewrite here can fail is by being unrepresentable — never by being wrong.

This is what separates `loops` from `reordering`: interchange, distribution and fusion *shuffle* iterations and must consult the dependences; everything here only *renames* them.
Several transformations in this module exist precisely to prepare a nest for a later reordering — skewing for interchange, splitting for distribution, normalization for everything.

## The Loop Normal Form

**Definition (loop normal form).** A `StructuredLoop` with induction variable `i` is in *normal form* if

1. it **starts at zero**: `init == 0`,
2. it has **unit positive stride**: `update == i + 1`,
3. its condition is **relational**: `i < bound` (possibly conjoined with further bounds), and
4. the induction variable's value **after the loop is closed-form**: `i == trip_count`, assigned explicitly after the loop.

In normal form the trip count *is* the bound, the iteration space is the integer interval $[0, \text{bound})$, and index expressions in the body carry the original stride and offset explicitly (`A[2*i + 10]`) where symbolic and polyhedral analyses can see them.

`LoopNormalFormPass` reaches the normal form loop by loop, with each step establishing the precondition of the next:

| Step | Transformation | Rewrites | Requires |
|---|---|---|---|
| 1 | `LoopShift` | `for (i = init; ...)` → `for (i = 0; ...)`, body uses `i + init` | integer init |
| 2 | `LoopUnitStride` | `i += s` → `i += 1`, condition and body use `s * i` | `init == 0` |
| 3 | `LoopConditionNormalize` | `i != N` → `i < N` (or `i > N` for stride $-1$) | $\lvert stride \rvert = 1$ |
| 4 | `LoopRotate` | stride $-1$ → stride $+1$, body uses `init + bound + 1 - i`; followed by another shift and condition normalization | stride $-1$, relational condition |
| 5 | `LoopIndvarFinalize` | assigns `i = trip_count` after the loop | normal form (steps 1–4) |

Steps 1, 2 and 4 do not rewrite the body directly: they record the original index (`__i_orig = i + init`) as an assignment on the loop's transition. `SymbolPropagation` then folds it into the body's index expressions, and dead code elimination removes the temporary.

`LoopRotate` deserves a second look, because it *reverses* the direction of the induction variable.
It is still order-preserving: the new iteration `i' = 1` executes the body for the original value `10`, `i' = 2` for `9`, and so on — the instances run in exactly the original order; only their *names* count up instead of down.

`LoopIndvarFinalize` is not an iteration-space rewrite at all, but it is part of the normal form for a reason: code after a loop often reads the induction variable, which creates a dependence from the *last* iteration to that read.
Replacing it with the closed form `trip_count` severs the dependence, which is one of the structural gates `AutoParallelization` checks before turning a `For` into a `Map`.

## Partitioning the Iteration Space

**`LoopSplit`** cuts a unit-stride loop at a symbolic point:

$$
[\text{init}, \text{bound}) \;=\; [\text{init}, p) \,\cup\, [p, \text{bound})
$$

The two loops run back to back, so the union is executed in the original order; either half is simply empty if $p$ lies outside the range.
Its main use is to make triangular nests distributable — e.g. splitting the `k` loop of a blocked LU factorization at the tile boundary, so that `LoopDistribute` can separate the two halves.

**`LoopPeeling`** targets loops with a compound condition — a constant-trip tile bound plus dynamic bounds, `for (k = k0; k < k0 + TK && k < N; k++)` — and makes a whole perfectly nested chain of them constant-trip and zero-based, so the nest can be fully unrolled or vectorized.
The dropped dynamic bounds are reinstated in one of two ways:

- **hoisted** (default): an outer `IfElse` runs the clean constant-trip nest when the whole tile is in bounds, and the original nest otherwise — every instance runs once, in one of the two versions;
- **predicated**: the constant-trip nest runs unconditionally and the innermost body is guarded by the dynamic bounds — the extra iterations are empty, so the non-empty instances keep their order. On GPUs the guard lowers to predication and keeps register tiles in registers.

## Strip-Mining: `StripMining` and `MultiLevelTiling`

`StripMining` splits one loop into a loop over tiles and a loop within a tile:

```c++
for (int i = 0; i < N; i++)          for (int t = 0; t < N; t += T)
    body(i);                    =>       for (int i = t; i < t + T && i < N; i++)
                                             body(i);
```

The remainder tile is handled by conjoining the original condition into the inner bound, so `N` need not be a multiple of `T`.
If the original loop is a `Map`, the tile loop stays a `Map` and the element loop becomes sequential.
`MultiLevelTiling` applies the same split repeatedly: $n$ tile sizes produce $n + 1$ loops, each tile size dividing the next outer one.

> **Note.** The name is deliberate: this is **strip-mining**, not tiling (it was called `LoopTiling` before; that name remains as an alias, and recorded `"LoopTiling"` transformations still replay). Tiling a *nest* — the classic blocked matrix multiplication — is strip-mining every loop and then *interchanging* the tile loops outward. The strip-mining is always legal; the interchange is a reordering and is checked by `reordering::LoopInterchange`.

`TilingPass` applies `StripMining` to a set of loops in two phases (check all, then apply all); the CUDA and ROCm offload schedulers use it to block loops for the grid.

## Skewing: Preparing for Interchange

`LoopSkewing` shifts the inner loop of a pair by a multiple of the outer induction variable:

```c++
for (i = 0; i < N; i++)                for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)      =>        for (j' = f*i; j' < M + f*i; j'++)
        body(i, j);                            body(i, j' - f*i);
```

Each instance `(i, j)` becomes `(i, j + f*i)`. The inner loop still runs in increasing order, so the transformation is order-preserving.
Its value lies in what it does to the distance vectors: a dependence $(d_i, d_j)$ becomes $(d_i, d_j + f \cdot d_i)$.

Take the classic wavefront `A[i][j] = A[i-1][j] + A[i][j-1]`, with distances $(1, 0)$ and $(0, 1)$.
Neither loop is parallel, and interchange alone yields nothing useful.
After skewing with $f = 1$ the distances are $(1, 1)$ and $(0, 1)$; after *then* interchanging, they become $(1, 1)$ and $(1, 0)$ — both carried by the new outer loop, leaving the inner loop free of carried dependences.
The skew is the legal, order-preserving half of that recipe; the interchange is the reordering half.

## Collapsing: `MapCollapse` and `CollapseToDepth`

`MapCollapse` flattens a perfect nest of `count` maps into a single map over the product space, recovering the original indices by division and modulo:

```c++
map (i in [0, N))                     map (k in [0, N*M))
    map (j in [0, M))         =>          body(k / M, k % M);
        body(i, j);
```

The lexicographic order of `(i, j)` is exactly the linear order of `k`, so a perfect collapse is order-preserving — and since the loops are `Map`s, order is irrelevant anyway.

The **imperfect** mode (`count == 2`, CUDA-kernel style) is the one exception in this module: it collapses an outer map whose body contains *several* sibling maps and other elements, guarding each sibling by its bound and running the other elements once per outer iteration. This *interleaves* siblings that previously ran one after another, and replicates the other elements across threads — so it carries its own data-dependency gate, rejecting inner-varying writes shared between body elements, write-write conflicts, and read-modify-writes on shared containers.

`CollapseToDepth` collapses a perfect map nest to one loop, or to two (an outer and an inner half). `CollapsePass` applies it in two phases; the OpenMP scheduler collapses to depth 1, the CUDA and ROCm schedulers to depth 2.

## Unrolling

`UnrollTransform` currently does not rewrite the loop: it marks a constant-trip loop with the `unroll` schedule property, which code generation lowers to `#pragma clang loop unroll(full)`.
Its main purpose is to let the compiler scalarize register tiles. A proper unrolling transformation — replicating the body, with a remainder loop — is planned to replace it.

## Layout

| Path | Contents |
|---|---|
| `loop_header.h` | `LoopHeader` (init, condition, update), `LoopSwap` (two nested loops and their proposed headers, shared with `reordering::LoopInterchange` and `tiles::ReductionBufferAnalysis`) |
| `transformations/` | normal form: `LoopShift`, `LoopUnitStride`, `LoopConditionNormalize`, `LoopRotate`, `LoopIndvarFinalize` |
| | partitioning: `LoopSplit`, `LoopPeeling` |
| | strip-mining: `StripMining`, `MultiLevelTiling` |
| | reindexing: `LoopSkewing` |
| | collapsing: `MapCollapse`, `CollapseToDepth` |
| | `UnrollTransform` |
| `passes/` | `LoopNormalFormPass`, `TilingPass`, `CollapsePass` |

Everything lives in the `sdfg::loops` namespace.

## Summary

- An **order-preserving** transformation maps old iterations to new ones bijectively and monotonically; by the Fundamental Theorem it preserves every dependence, so its legality is purely a question of *representability*.
- The **loop normal form** — start at zero, unit positive stride, relational condition, closed-form final induction value — is reached by `LoopNormalFormPass` through shift, unit-stride, condition normalization, rotation and finalization, each enabling the next.
- **Splitting**, **peeling**, **strip-mining**, **skewing** and **collapsing** reshape the iteration space without reordering it; most of them exist to enable a later reordering (distribution, interchange) or a schedule (tiling for GPUs, collapsing for OpenMP and GPU grids).
- The only exception is **imperfect map collapse**, which interleaves sibling maps and therefore checks data dependencies itself.
