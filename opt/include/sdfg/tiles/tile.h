#pragma once

#include <optional>
#include <string>
#include <vector>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/layout.h"

namespace sdfg {
namespace analysis {
class AnalysisManager;
}
namespace tiles {

class LocalityPlan; // produced by Tile::placement, see locality.h

/**
 * @file tile.h
 * @brief Schedule-aware tile vocabulary: a @ref Tile is a @ref Layout (geometry
 *        from MemoryLayoutAnalysis) partitioned over the enclosing parallel loop
 *        nest (@ref TileAxis) and classified into a required memory @ref Space.
 *
 * Pipeline:  @ref AxisSchedule::classify → @ref TileAxis::enclosing → @ref Tile
 *            → @ref Tile::placement (a @ref LocalityPlan) → storage @ref Space.
 */

/// Cooperation scope of a parallel axis, coarsest to finest — target-neutral tiers
/// (OpenCL/SYCL vocabulary). Each target maps its schedule onto them via
/// @ref AxisSchedule::classify: GPU grid / CPU threads -> Device, GPU block /
/// Tenstorrent core-workers -> Group, GPU warp / wavefront -> Subgroup.
enum class Level { Device, Group, Subgroup };

/// Memory that backs cooperation at a level:
/// Device/CPU -> Global (atomics/heap), Group -> Shared, Subgroup/none -> Register.
enum class Space { Global, Shared, Register };

/// Whether the tile is per-iteration-private or shared across a parallel axis.
enum class Role { Private, Cooperative };

/// The memory space backing cooperation at @p level in the canonical scratchpad
/// hierarchy (Device->Global, Group->Shared, Subgroup->Register): the default a
/// scratchpad target reuses. A target may remap it via @ref TileTarget::space
/// (e.g. a flat CPU sends every level to Global, having no scratchpad).
Space default_space(Level level);

/// Target-neutral cooperation facts derived from one loop's schedule.
class AxisSchedule {
    Level level_ = Level::Device;
    Space space_ = Space::Global; ///< the memory backing this axis's cooperation (target-assigned)
    bool has_scratchpad_ = false; ///< the owning target exposes on-chip cooperative tiers below global
    unsigned spatial_axis_ = 0; ///< the parallel grid dimension of this axis: 0=X, 1=Y, 2=Z
    symbolic::Integer parallel_size_ = symbolic::integer(0); ///< 0 when unknown (CPU)
    bool needs_sync_ = false;

public:
    AxisSchedule() = default;
    AxisSchedule(
        Level level,
        Space space,
        bool has_scratchpad,
        unsigned spatial_axis = 0,
        symbolic::Integer parallel_size = symbolic::integer(0),
        bool needs_sync = false
    );

    /// Classify a loop's schedule, or `std::nullopt` for a sequential loop (which
    /// does not shape storage). New offload schedule types extend this factory.
    static std::optional<AxisSchedule> classify(const structured_control_flow::ScheduleType& sched);

    /// The cooperation @ref Level a target assigns to @p sched, or `std::nullopt`
    /// when no registered target treats it as a scratchpad (device-parallel)
    /// schedule (a sequential or host-thread loop). Neutral schedule test:
    /// `== Level::Group` means group-cooperative, `== Level::Device` means device-wide.
    static std::optional<Level> classify_level(const structured_control_flow::ScheduleType& sched);

    /// True when @p sched is a group-level schedule whose parallel threads can drive
    /// a cooperative shared-memory staging copy (a genuine offload schedule).
    /// Distinguishes real offload block schedules from legacy fused whole-kernel
    /// schedules, which cooperate at group level but cannot host a separate copy Map.
    static bool drives_cooperative_copy(const structured_control_flow::ScheduleType& sched);

    bool has_scratchpad() const { return has_scratchpad_; }
    Level level() const { return level_; }
    Space space() const { return space_; }
    unsigned spatial_axis() const { return spatial_axis_; }
    const symbolic::Integer& parallel_size() const { return parallel_size_; }
    bool needs_sync() const { return needs_sync_; }
};

/// One enclosing parallel loop of a tile: its schedule facts plus the tile-role
/// (Private when its indvar addresses the tile base, Cooperative otherwise).
class TileAxis {
    symbolic::Symbol indvar_;
    Role role_ = Role::Private;
    AxisSchedule schedule_;
    symbolic::Expression init_ = symbolic::integer(0);
    symbolic::Integer stride_ = symbolic::integer(1);

public:
    TileAxis() = default;
    TileAxis(
        symbolic::Symbol indvar,
        Role role,
        AxisSchedule schedule,
        symbolic::Expression init = symbolic::integer(0),
        symbolic::Integer stride = symbolic::integer(1)
    );

    /// The enclosing parallel axes of a tile with the given @p bases (min index per
    /// dim): an axis is Cooperative when its indvar addresses no base, Private
    /// otherwise; sequential loops are skipped. Innermost-first. The single source
    /// of the schedule classification.
    static std::vector<TileAxis>
    enclosing(structured_control_flow::StructuredLoop& loop, const symbolic::MultiExpression& bases);

    const symbolic::Symbol& indvar() const { return indvar_; }
    Role role() const { return role_; }
    const AxisSchedule& schedule() const { return schedule_; }
    const symbolic::Expression& init() const { return init_; }
    const symbolic::Integer& stride() const { return stride_; }
    bool cooperative() const { return role_ == Role::Cooperative; }
};

/// A container's staged region: `source` maps a tile coordinate to the global
/// element, `axes` is the thread/value partition, and the read/write flags give
/// the copy direction. The buffer placement and copy atom are derived from this.
class Tile {
    std::string container_;
    Layout source_;
    std::vector<TileAxis> axes_; ///< enclosing parallel axes, innermost-first
    bool reads_ = false;
    bool writes_ = false;

public:
    Tile() = default;
    Tile(std::string container, Layout source, std::vector<TileAxis> axes, bool reads, bool writes);

    const std::string& container() const { return container_; }
    const Layout& source() const { return source_; }
    const std::vector<TileAxis>& axes() const { return axes_; }
    bool reads() const { return reads_; }
    bool writes() const { return writes_; }

    bool cooperative() const;
    std::vector<TileAxis> cooperative_axes() const;
    std::vector<TileAxis> private_axes() const;

    /// The memory space the tile must live in: the coarsest cooperative axis's
    /// space (Global > Shared > Register), or Register when fully private.
    Space required_space() const;

    /// The tile's placement in the enclosing parallel nest at @p loop — its axes
    /// plus loop-context flags, the basis for deriving storage and synchronization.
    LocalityPlan placement(structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager)
        const;
};

} // namespace tiles
} // namespace sdfg
