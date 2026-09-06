#pragma once

#include <optional>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace tiles {

/**
 * @file tile_target.h
 * @brief The complete target-specific contract of the tile algebra.
 *
 * Everything else in the tiles API is target-neutral: @ref Layout, @ref Tile,
 * @ref LocalityPlan, @ref PackedBuffer, @ref TiledCopy, and the @ref Level /
 * @ref Space lattices. A backend (CUDA, ROCm, OpenMP, Tenstorrent, ...) makes its
 * schedules and memory legible to that neutral core by implementing a handful of
 * per-target facts:
 *
 *   1. how its loop schedules cooperate         \-> @ref classify (a schedule -> @ref AxisSchedule),
 *   2. which memory tier backs each cooperation level \-> @ref space (a @ref Level -> @ref Space),
 *   3. whether a schedule can drive a cooperative staging copy \-> @ref supports_cooperative_staging,
 *   4. which concrete buffer realizes a memory tier \-> @ref storage_type (a @ref Space -> StorageType),
 *   5. its SIMD lane width for bank-conflict avoidance \-> @ref lane_width.
 *
 * (@ref has_scratchpad is derived from @ref space, not implemented.) These are the
 * *only* places the tile algebra needs to know a target exists. A @ref TileTarget
 * replaces the hard-coded `"CUDA_Offload"` / `"ROCM_Offload"` / `NV_Shared` /
 * `gpu_warp_size` checks that would otherwise scatter through `AxisSchedule::classify`
 * and `LocalStorage`. New targets (including externally linked ones) plug in by
 * implementing this interface and registering it; no edit to the tiles core is
 * required.
 */
class TileTarget {
public:
    virtual ~TileTarget() = default;

    /// Classify one loop's @p sched into target-neutral cooperation facts
    /// (@ref AxisSchedule: level, spatial axis, parallel size, sync), or
    /// `std::nullopt` when the schedule does not shape storage (a sequential loop).
    /// Replaces the target-string switch inside `AxisSchedule::classify`.
    virtual std::optional<AxisSchedule> classify(const structured_control_flow::ScheduleType& sched) const = 0;

    /// The memory @ref Space backing cooperation at each @ref Level on this target:
    /// a GPU maps `Group -> Shared`, `Subgroup -> Register`; a flat CPU maps every
    /// level to `Global`; an exotic scratchpad target maps whichever levels it
    /// materializes on-chip. `classify` stamps each axis with `space(its
    /// level)`. This is the single per-level, per-target memory fact the tile
    /// algebra reads.
    virtual Space space(Level level) const = 0;

    /// Whether this target exposes any on-chip cooperative tier below global memory
    /// (a scratchpad hierarchy), derived from @ref space. `classify` stamps
    /// it onto every axis so the tile algebra can tell a device-wide axis that sits
    /// atop a scratchpad (GPU grid) from a flat host axis, without naming any target.
    bool has_scratchpad() const {
        return space(Level::Group) != Space::Global || space(Level::Subgroup) != Space::Global;
    }

    /// Whether @p sched can host a block-cooperative shared-memory staging copy
    /// driven by its own parallel threads (i.e. a genuine offload schedule), as
    /// opposed to a legacy fused whole-kernel schedule that cooperates at group
    /// level but cannot carry a separate copy Map. Distinguishes the two where
    /// `classify` alone maps both to @ref Level::Group.
    virtual bool supports_cooperative_staging(const structured_control_flow::ScheduleType& sched) const = 0;

    /// The concrete buffer that realizes an abstract cooperation @ref Space on this
    /// target: e.g. `Shared -> NV_Shared` / `TT_L1`, `Global -> NV_Global`,
    /// `Register -> ` a thread-private stack/register block. Replaces the
    /// `Space`-to-`StorageType` switch in `LocalStorage`.
    virtual types::StorageType storage_type(Space space) const = 0;

    /// The SIMD lane / subgroup width, used only to keep a subgroup's cooperative
    /// stores bank-conflict-free (32 on NVIDIA, 64 on CDNA, 1 where there is no
    /// SIMD cooperation). Replaces `gpu::gpu_warp_size` in the padding heuristic.
    virtual unsigned lane_width() const = 0;
};

} // namespace tiles
} // namespace sdfg
