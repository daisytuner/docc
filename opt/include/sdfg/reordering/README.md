# Loop Reordering: Changing *When* Iterations Run, Not *What* They Compute

A loop nest fixes two things at once: *which* statement instances execute, and *in what order*.
The first is the meaning of the program; the second is mostly an accident of how it was written.
Hardware cares a great deal about the second.
Walking a row-major matrix column by column touches a new cache line on every access, while walking it row by row reuses each line for many consecutive elements.

```c++
void scale(float* A, int N, int M) {
    for (int j = 0; j < M; j++) {        // columns outside ...
        for (int i = 0; i < N; i++) {    // ... rows inside: stride M between accesses
            A[i * M + j] *= 2.0f;
        }
    }
}
```

Swapping the two loops computes exactly the same values — every `A[i][j]` is still scaled once — but the inner loop now walks contiguous memory.
Nothing was added, nothing was removed; only the *order* changed.
This module collects the transformations that do precisely that, and the analysis that decides when it is safe.

## Reordering Transformations

**Definition (reordering transformation).** A program transformation is a *reordering transformation* if it only changes the order in which statement instances execute, without adding or removing any instance.
*(Allen & Kennedy, "Optimizing Compilers for Modern Architectures", Ch. 2.)*

The definition draws a sharp line through the loop transformations of this library:

| Changes execution order? | Transformations | Legality question |
|---|---|---|
| **yes** — reordering | `LoopInterchange`, `LoopDistribute`, `MapFusion` / `LoopFusionPass` | *does the new order still respect every dependence?* |
| **no** — order-preserving | `LoopShift`, `LoopSkewing`, `StripMining`, `LoopSplit`, `LoopPeeling`, ... | *is the loop shape supported?* |

Skewing and strip-mining are often thought of as "schedule" transformations, but on their own they only re-index the iteration space: the instances still execute in the original lexicographic order.
They become order-changing only when *combined* with an interchange — e.g. tiling = strip-mining + interchange — and it is the interchange step whose legality must be checked.
That is why only interchange, distribution and fusion live here.

> **Intuition.** An order-preserving transformation renames the iterations. A reordering transformation shuffles them. Only shuffling can break a program, and it breaks it in exactly one way: some value is read before it was written, or overwritten before it was read.

## The Fundamental Theorem of Dependence

A **dependence** links two statement instances that touch the same memory cell, at least one of them a write (see the [parallelization README](../parallelization/README.md) for the RAW / WAW / WAR taxonomy).
The earlier instance is the *source*, the later one the *sink*.

**Theorem (Fundamental Theorem of Dependence).** A reordering transformation preserves the meaning of a program if it preserves the relative order of the source and sink of every dependence.

The theorem turns legality into bookkeeping: enumerate the dependences, apply the proposed reordering to each, and check that no sink now runs before its source.
Every transformation in this module is an instance of that check, specialized to its own shape of reordering.

### Dependences as distance vectors

Inside a loop nest $(i_1, \dots, i_n)$ an instance is identified by its iteration vector, and a dependence from iteration $a$ to iteration $b$ by its **distance vector** $d = b - a$.
In the original program the source runs first, so every distance vector is **lexicographically positive**: its first non-zero entry is positive.

A reordering is legal if and only if every distance vector *remains* lexicographically positive after the transformation is applied to it.

`LoopCarriedDependencyAnalysis` (in `analysis/`) is the producer of those vectors.
For a loop `L` it returns per-pair records `(writer, reader, type, Δ)`, where the delta set `Δ` is the ISL set of all distance vectors over the dimensions of the nest, computed symbolically from the access subsets.
An empty `Δ` proves independence; a non-empty `Δ` describes *every* distance at once, so legality checks become ISL emptiness queries rather than enumerations.
The analysis also recognizes reductions (`detect_reductions`), whose carried dependencies are reorderable by associativity and commutativity.
`parallelization` consumes the same analysis to decide whether a loop may become a `Map` — parallelization is, after all, the most aggressive reordering of all: *any* order.

## Loop Interchange

`LoopInterchange` swaps a perfectly nested pair of loops.
On distance vectors, it swaps the two corresponding entries: $(d_{outer}, d_{inner}) \mapsto (d_{inner}, d_{outer})$.

Consider a stencil whose dependence points "down-left":

```c++
for (int i = 1; i < N; i++)
    for (int j = 0; j < M - 1; j++)
        A[i][j] = A[i - 1][j + 1] + 1.0f;   // reads the value written at (i-1, j+1)
```

The read at `(i, j)` consumes the write at `(i-1, j+1)`, so the distance is $(1, -1)$ — lexicographically positive, as it must be.
After interchange it becomes $(-1, 1)$: negative in its first entry.
The interchanged nest would read `A[i-1][j+1]` *before* iteration `(i-1, j+1)` has produced it. Interchange is illegal here.

The check in `can_be_applied` follows directly:

- Dependences **carried by the inner loop** ($d_{outer} = 0$) stay within one outer iteration, and interchange keeps their relative order.
- Dependences **carried by the outer loop** ($d_{outer} > 0$) stay lexicographically positive iff $d_{inner} \geq 0$.
  This is tested by projecting `Δ` onto the inner dimension and asking ISL whether it intersects $\{ x < 0 \}$.
- **Reductions** (recognized by LCDA, or carried by an inner `Reduce`) are exempt: any accumulation order is valid.
- **Privatizable scalars** — loop-local scalar temporaries without a carried flow dependence — are exempt: each iteration may get its own copy.
- If either loop is a **`Map`**, the dependence check is skipped: a `Map` asserts that its iterations are independent.

Beyond dependences, interchange imposes structural requirements: the outer loop's body must be exactly the inner loop (a *perfect* nest), the body must not contain views or moves, and affected GPU reduction buffers (`tiles::ReductionBufferAnalysis`) must remain representable.

When the inner bounds depend on the outer induction variable — a triangular nest such as `for i: for j in [i, N)` — the loop headers cannot simply be swapped.
For `For`-`For` nests whose inner bounds are `min`/`max` of affine terms with non-negative coefficients, interchange performs a **Fourier-Motzkin** elimination: the new outer range spans the first and last outer iteration, and each affine bound is inverted onto the outer loop's stride lattice.
`proposal()` computes these new headers without modifying the graph.

## Loop Distribution

`LoopDistribute` (loop fission) splits a loop with several children into several loops over the same iteration space:

```c++
for (int i = 0; i < N; i++) {          for (int i = 0; i < N; i++)
    A[i] = f(i);                  =>       A[i] = f(i);
    B[i] = A[i] * 2.0f;                for (int i = 0; i < N; i++)
}                                          B[i] = A[i] * 2.0f;
```

Before distribution, the instances interleave: `A[0], B[0], A[1], B[1], ...`.
After it, *all* instances of the first statement run before *any* instance of the second.
Every dependence from the second piece back to the first (a sink in an earlier piece, a source in a later one) would be reversed.

`LoopDistribute` performs an in-place three-way split around a chosen child — *prefix*, *center* (the child), and *suffix* — preserving program order between the groups.
For each loop-carried pair from LCDA:

- pairs whose writer and reader land in the **same group** are untouched — they stay in one shared loop body;
- cross-group **WAW** pairs are safe: groups keep their program order, so the last writer of every cell is unchanged;
- cross-group **RAW** pairs are rejected.

In addition, a **scalar** that is written in one piece and read in another *within the same iteration* blocks distribution whenever that scalar is also loop-carried: after the split, the reader would see the writer's *last* iteration instead of the matching one.
Distributing such loops would require scalar expansion first.

> **Note.** Rejecting every cross-group RAW is conservative. A RAW whose writer lives in an *earlier* group than its reader keeps its source before its sink after distribution; it stays correct as long as the cell is not overwritten again before the read (no accompanying WAW on it). Only pairs whose reader lives in an earlier group than the writer are reversed by the split.

`PerfectLoopDistributionPass` applies `LoopDistribute` bottom-up (deepest loops first) to every loop whose body mixes blocks and sub-loops, peeling them apart until the nests are perfect — the shape interchange requires.

## Loop Fusion

Fusion is the inverse of distribution: two adjacent loops over compatible iteration spaces are merged into one, so that a value produced in iteration `i` of the first loop is consumed while it is still in a register or cache.

```c++
for (int i = 0; i < N; i++)                for (int i = 0; i < N; i++) {
    T[i] = A[i] + 1.0f;            =>          T[i] = A[i] + 1.0f;
for (int i = 0; i < N; i++)                    B[i] = T[i] * 2.0f;
    B[i] = T[i] * 2.0f;                    }
```

Fusion interleaves what was separated, so the dependences at risk are those between the two loops.
A dependence from producer iteration $a$ to consumer iteration $b$ survives fusion iff $a \leq b$ — the consumer may only use values the fused loop has already produced.
A consumer reading `T[i + 1]` is the classic **fusion-preventing** dependence.

Fusion in this module does not go through LCDA; it reasons directly about producer writes and consumer reads (`fusion/`), and comes in two strategies:

- **By domain.** Both loops (or both stacks of perfectly nested `Map`s) have exactly equal iteration domains, and every overlapping access uses exactly the same subset. Every dependence then has distance $0$, and the loop bodies can simply be concatenated. This also fuses loops whose bodies contain further, arbitrary loops.
- **By access** (`LoopFusionByAccessWorker`). When domains or subsets differ, each consumer read is solved for the producer iteration that wrote it (`solve_subsets`), and the producer computation is inlined at the read — scalarizing the intermediate. This works across differing iteration domains, at the price of possibly recomputing producer values.

> **Note.** By-access fusion with recomputation is, strictly speaking, *not* a reordering transformation: it may execute some producer instances more than once. Its legality argument is therefore "every read is covered by a matching write" rather than the Fundamental Theorem.

`MapFusion` is the single-pair transformation, supporting three patterns: producer-into-consumer for perfect nests, consumer-into-producer when the producer is not perfectly nested, and the reverse when only the consumer is not.
With `allow_init_hoist`, it additionally handles initializing a reduction accumulator: the initializing `Map` is hoisted to the reduction's outer parallel band instead of being inlined element-by-element.

`LoopFusionPass` drives fusion over a whole SDFG: it walks each sequence in execution order with a sliding window of three, tries by-domain fusion first and falls back to by-access, and keeps a `FusionLoopCandidate` cache up to date across modifications. `MapFusionPass` is the simpler visitor-based driver over consecutive `(Map, StructuredLoop)` pairs.

## Choosing an Order: `StrideMinimization`

Legality says which orders are *allowed*; `StrideMinimization` picks a *good* one.
For each loop-tree path it:

1. asks `LoopInterchange::can_be_applied` which **adjacent swaps** are legal;
2. enumerates all permutations of the nest and keeps those reachable by a sequence of legal adjacent swaps (a bubble-sort decomposition, `is_admissible`);
3. scores each admissible permutation by the **stride** each loop variable induces in the nest's accesses — the position of the subset dimension it indexes, counted from the innermost dimension — and prefers the permutation whose innermost loops have the smallest maximal stride;
4. applies the winning permutation as a sequence of interchanges.

## Pipelines

| Entry point | Runs |
|---|---|
| `passes::normalization::loop_normalization()` | `PerfectLoopDistributionPass`, then `StrideMinimization` |
| `passes::normalization::stride_minimization()` | `StrideMinimization` |
| `passes::normalization::map_fusion(...)` | `MapFusionPass` with block fusion and dead-data / dead-CFG cleanup |
| `passes::normalization::normalize(sdfg, enable_fusion)` | without fusion: `loop_normalization()`; with fusion: stride minimization, `LoopFusionPass`, cleanup, and a final `map_fusion` run with init hoisting |

Distribution and fusion pull in opposite directions; the pipelines pick one per run rather than letting them undo each other.

## Layout

| Path | Namespace | Contents |
|---|---|---|
| `analysis/` | `sdfg::reordering` | `LoopCarriedDependencyAnalysis` |
| `transformations/` | `sdfg::reordering` | `LoopInterchange`, `LoopDistribute` |
| `passes/` | `sdfg::reordering` | `StrideMinimization`, `PerfectLoopDistributionPass` |
| `fusion/` | `sdfg::reordering::fusion` | fusion candidates and `LoopFusionByAccessWorker` |
| `fusion/transformations/` | `sdfg::reordering::fusion` | `MapFusion` |
| `fusion/passes/` | `sdfg::reordering::fusion` | `LoopFusionPass`, `MapFusionPass` |

## Summary

- A **reordering transformation** changes the order of statement instances without adding or removing any; interchange, distribution and fusion are reordering transformations, while shifting, skewing and strip-mining are not.
- By the **Fundamental Theorem of Dependence**, a reordering is legal iff every dependence keeps its source before its sink — equivalently, every distance vector stays lexicographically positive.
- `LoopCarriedDependencyAnalysis` provides those distance vectors as ISL **delta sets**, shared with `parallelization`.
- **Interchange** swaps two distance entries and is legal when no outer-carried dependence has a negative inner distance; **distribution** serializes the pieces of a body and must not reverse a flow between them; **fusion** interleaves two loops and must not make a consumer read ahead of its producer.
- `StrideMinimization` searches the legal permutations for the one with the most contiguous innermost accesses, and the normalization pipelines combine these building blocks into a loop normal form.
