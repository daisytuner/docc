# Scheduling: Deciding *Where* and *How* Parallel Work Runs

By the time a program reaches scheduling, the earlier modules have already decided *what* may run in parallel.
[Parallelization](../parallelization/README.md) has turned independent `For` loops into `Map`s and accumulations into `Reduce`s, and [reordering](../reordering/README.md) and [loops](../loops/README.md) have reshaped the nests.
What is still missing is the binding to a machine: does this `Map` become an OpenMP worksharing loop, a CUDA grid, a SIMD loop, or nothing at all?

```c++
for (int i = 0; i < N; i++)          // Map: iterations are independent
    for (int j = 0; j < M; j++)      // Map
        C[i][j] = A[i][j] + B[i][j];
```

The nest above means the same thing on every target.
On a CPU we would collapse `i` and `j` and hand the iterations to OpenMP threads; on a GPU we would map them onto a two-dimensional grid of blocks and threads; on a SIMD unit the inner loop would become vector lanes.
Scheduling is the step that makes this choice and rewrites the SDFG accordingly.
It never changes *what* is computed — only where and in which form it executes.

## Two Scopes: Loop Nests and SDFG

A scheduling decision can be taken with two very different views of the program.
Consider a function with three parallel loop nests:

```c++
for (int i = 0; i < N; i++)          // nest 1: Map
    T[i] = A[i] * A[i];
for (int i = 0; i < N; i++)          // nest 2: Map
    B[i] = T[i] + 1.0f;
float s = 0.0f;
for (int i = 0; i < N; i++)          // nest 3: Reduce
    s += B[i];
```

Looked at one by one, each nest is a perfectly good GPU kernel.
Looked at as a whole, the picture changes:

- if nests 1 and 2 run on the device, `T` never has to leave it — but only if both are placed there;
- if nest 3 stays on the host, `B` has to be copied back, and the transfer may cost more than the kernels saved;
- for small `N`, three kernel launches plus transfers lose against running everything on the CPU;
- fusing nests 1 and 2 removes `T` altogether, which changes the trade-off again.

None of this is visible from inside a single nest.
It depends on how the nests share data, and every decision changes the cost of the others.

This splits scheduling into two problems:

- **Loop scheduling** binds *one* loop nest to *one* target: which loop becomes the thread or block dimension, which becomes SIMD lanes, how many levels to collapse.
  It needs only local properties — trip counts, perfect nesting, dependences, side effects — and its decisions are independent across nests, which makes it cheap, composable and predictable.
- **SDFG scheduling** decides for the *whole* function: which nest runs on which target, where intermediate containers live, which data moves when, and which nests should be fused or kept apart.
  It reasons about relations *between* nests — producer–consumer edges, container lifetimes, transfer volume against compute — and its decisions are coupled.

The two are not alternatives but layers: the global schedule decides placement and data, and loop scheduling then maps each nest onto the target it was given.

The same split appears in scheduling languages such as Halide (Ragan-Kelley et al., PLDI 2013), which separates per-stage loop directives from the choice of where a producer is computed and stored relative to its consumers, and Tiramisu (Baghdadi et al., CGO 2019), which separates computation scheduling from data and communication management.
docc does not expose a scheduling language: a schedule is applied as ordinary SDFG transformations (`OMPTransform`, `CUDATransform`, ...).
It can still be captured as data — with a `transformations::Recorder` attached, every applied transformation is recorded as a replayable descriptor.

## Scheduling for a Target

Every target (`docc::target::DoccTarget`) declares how it schedules an SDFG in its `scheduling`, and `sdfg::scheduling::schedule` ([target_scheduling.h](target_scheduling.h)) runs it, in this order:

1. the target's `SDFGScheduler` brings the whole SDFG into the normal form its loop schedulers expect — a priori loop nest normalization ([a_priori_loop_nest_normalization.h](a_priori_loop_nest_normalization.h)) for CPU, Tenstorrent and ET targets, nothing ([passthrough_scheduler.h](passthrough_scheduler.h)) for GPU targets;
2. the target's sched-time library node mapping;
3. remote tuning, if requested;
4. the target's loop schedulers, through the `LoopSchedulingPass`.

The `SDFGScheduler` must be deterministic: the tuning server runs it as well.
