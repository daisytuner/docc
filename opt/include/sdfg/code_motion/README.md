# Code Motion: Running a Subtree at Another Point

Code motion changes *where* a subtree of the structured control flow runs, not what it computes:
a child moves to another position in its sequence, or out of a loop to run once instead of in every iteration.
Unlike the loop reorderings in `reordering/`, the subtree's instances stay the same up to repetition; only their position relative to their neighbours changes.

## Transformations

| Transformation | Moves | Legal if |
|---|---|---|
| `MoveChild(child, position)` | a child of a sequence to `position` in it | the child and the siblings it passes are movable, and the child depends on none of them |
| `HoistChild(loop, child, after)` | a child of a loop's body before (or after) the loop | the loop runs at least once, the child does not use the counter or read what it writes, and no other child writes what it accesses in any pair of iterations; only children after it in the body may read what it writes, and only when moving before the loop |

A subtree is *movable* if it transfers no control (return, or break and continue outside a while loop within) and calls no library node with side effects other than its accesses: a library node with side effects may move if it has memlets and describes every pointer it takes (`LibraryNode::pointer_use_creates_side_effects`), as allocations, frees and tile copies do; one without memlets, as a barrier, stays. A pointer a library node invalidates, as a free does, counts as written all over.

## Dependences Between Subtrees

`may_conflict(first, second, scope, kinds)` decides whether two subtrees depend on one another by flow (the first writes what the second reads), anti (the first reads what the second writes) or output (both write) dependences.
For each container the kinds relate, the elements both access must be disjoint:

- The elements are the tiles `MemoryLayoutAnalysis` bounds them by: per memlet in a block, per tile group in any other subtree.
- A container is accessed anywhere if a subtree uses it as a symbol (in a subset, condition, loop header or assignment), views, moves or captures its pointer, or touches it through a memlet no tile covers.
- Whether a subtree reads or writes a container follows the users and the library nodes' pointer metadata.
- The tiles are compared with ISL. Symbols constant in `scope` take the same value in both subtrees, so `B[0:N]` and `B[N:2N]` are disjoint; the others take any value in each, so marking a loop's counter non-constant compares the subtrees across all pairs of iterations.
