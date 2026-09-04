#pragma once

#include <functional>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/buffer_layout.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace tiles {

/// Who provides the visibility fence around the copy.
enum class SyncPolicy {
    None, ///< caller owns the fences (e.g. a pipeline's commit/wait/drain)
    SingleStage, ///< leading + trailing block barrier around the copy
};

/// Which way the tile moves between the flat (global) side and the packed buffer.
enum class CopyDirection {
    In, ///< read the global tile into the local buffer (copy-in / stage)
    Out, ///< write the local buffer back to the global tile (copy-out / writeback)
};

/// The concrete SDFG containers/types a @ref TiledCopy moves between. The
/// `global` side is a flat pointer addressed *linearly* (`plan.src.apply_coords`);
/// the `buffer` side is a dense nested-array addressed *multi-dimensionally* (one
/// index per tile axis), so clang recovers the per-axis strides and vectorizes.
struct CopyContainers {
    std::string global; ///< flat (source) pointer container
    std::string buffer; ///< packed local-buffer container (nested array)
    const types::IType* global_type; ///< pointer type for the linear global memlet
    const types::IType* buffer_type; ///< nested-array type for the multi-dim buffer memlet
};

/// Per-element legality predicate: given the tile coordinate (`plan.src` mode
/// indices, outermost-first) return a guard that must hold for the element to be
/// copied. `SymEngine::boolTrue` (or an empty function) means "always copy". The
/// caller owns the policy — e.g. dropping conjuncts provably true under its
/// assumptions so the interior copy vectorizes.
using BoundaryGuard = std::function<symbolic::Condition(const std::vector<symbolic::Expression>&)>;

/// How the copy's iteration space is swept.
enum class Coverage {
    PerMode, ///< one nested Map per src mode — dense, per-axis vectorizable
    Flat, ///< one Map over the linearized tile, row-major delinearized; a GPU
          ///< offload @c schedule then splits that flat range across the lanes
          ///< (the cooperative copy), independent of tile rank
};

/**
 * @brief Emit the copy's map nest + guarded element move *into* an existing scope,
 *        without creating a scope or any barriers.
 *
 * Emits one nested @c Map per @c plan.src mode (or a single flat map for
 * @ref Coverage::Flat) whose body moves one element: the global address is
 * `plan.src.apply_coords(coords)` (linear), the buffer address
 * `dst_buffer.subset(slot_indices, coords)` (per-axis, so the type is preserved).
 *
 * @p schedule is placed on every coverage map (`nullptr` = sequential; a GPU
 * offload schedule makes the copy cooperative). @p slot_indices are the per-thread
 * buffer-slot indices for a mixed tile. @p guard skips out-of-bounds elements.
 * Callers owning placement/fencing drive this directly; @ref emit wraps it.
 */
void emit_into(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const TiledCopy& plan,
    const CopyContainers& containers,
    CopyDirection direction,
    const PackedBuffer& dst_buffer,
    const structured_control_flow::ScheduleType* schedule = nullptr,
    const std::vector<symbolic::Expression>& slot_indices = {},
    const BoundaryGuard& guard = {},
    Coverage coverage = Coverage::PerMode
);

/// Materialize a @ref TiledCopy into the SDFG before @p before: @ref emit_into in
/// a fresh scope with optional @p sync barriers. @p direction picks copy-in vs
/// copy-out, @p guard optionally wraps the body in an out-of-bounds `if`.
void emit(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    structured_control_flow::ControlFlowNode& before,
    const TiledCopy& plan,
    const CopyContainers& containers,
    CopyDirection direction,
    SyncPolicy sync,
    const BoundaryGuard& guard = {}
);

} // namespace tiles
} // namespace sdfg
