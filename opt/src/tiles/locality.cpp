#include "sdfg/tiles/locality.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/tiles/tile.h"

namespace sdfg {
namespace tiles {

bool LocalityPlan::inside_scratchpad_scope() const {
    for (const auto& a : axes_)
        if (a.schedule().has_scratchpad()) return true;
    return false;
}

bool LocalityPlan::has_scratchpad_cooperative() const {
    for (const auto& a : axes_)
        if (a.schedule().has_scratchpad() && a.cooperative()) return true;
    return false;
}

bool LocalityPlan::has_cooperative_at(Level level) const {
    for (const auto& a : axes_)
        if (a.schedule().has_scratchpad() && a.cooperative() && a.schedule().level() == level) return true;
    return false;
}

bool LocalityPlan::has_global_cooperative() const {
    for (const auto& a : axes_)
        if (!a.schedule().has_scratchpad() && a.cooperative()) return true;
    return false;
}

std::vector<TileAxis> LocalityPlan::private_axes() const {
    std::vector<TileAxis> out;
    for (const auto& a : axes_)
        if (a.schedule().has_scratchpad() && !a.cooperative()) out.push_back(a);
    return out;
}

std::vector<TileAxis> LocalityPlan::cooperative_axes() const {
    std::vector<TileAxis> out;
    for (const auto& a : axes_)
        if (a.schedule().has_scratchpad() && a.cooperative()) out.push_back(a);
    return out;
}

structured_control_flow::StructuredLoop* find_block_scheduled_descendant(
    structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(desc);
        if (!sl) {
            continue;
        }
        if (AxisSchedule::drives_cooperative_copy(sl->schedule_type())) {
            return sl;
        }
    }
    return nullptr;
}

LocalityPlan LocalityPlan::analyze(
    structured_control_flow::StructuredLoop& loop,
    const std::vector<TileAxis>& axes,
    analysis::AnalysisManager& analysis_manager
) {
    LocalityPlan plan;
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    plan.loop_is_outermost_ = loop_analysis.is_outermost_loop(&loop);
    plan.loop_has_scratchpad_ = AxisSchedule::classify_level(loop.schedule_type()).has_value();
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(desc);
        if (sl && AxisSchedule::classify_level(sl->schedule_type()).has_value()) {
            plan.has_scratchpad_descendant_ = true;
            break;
        }
    }

    plan.axes_ = axes;

    // Enclosing-scope cooperative staging: the localized loop is itself a scratchpad
    // map with no enclosing parallel context, and a group-scheduled loop in its body
    // consumes the tile. It is staged once per group into shared and reused by every
    // (sibling) consumer below.
    if (plan.axes_.empty() && plan.loop_has_scratchpad_ &&
        find_block_scheduled_descendant(loop, analysis_manager) != nullptr) {
        plan.enclosing_cooperative_ = true;
    }

    return plan;
}

std::optional<Space> LocalityPlan::required_space(bool container_written) const {
    // A cooperative CPU-parallel axis means the tile is invariant across the parallel
    // threads (a shared operand, e.g. BLIS ~B). CPU has no shared staging, but a
    // read-only tile is safe by replication: each thread stages its own private copy
    // of the identical tile (redundant packing, no sharing). A written cooperative
    // tile is a genuine cross-thread reduction/race and must decline.
    if (has_global_cooperative()) {
        if (container_written) {
            return std::nullopt;
        }
        return Space::Register;
    }
    // Enclosing-scope staging: a per-block shared row loaded once, reused by the
    // block consumers below. A cooperative write is a reduction (Reduce owns it).
    if (enclosing_cooperative_) {
        if (container_written) {
            return std::nullopt;
        }
        return Space::Shared;
    }
    if (has_scratchpad_cooperative()) {
        // A cooperative write across threads is a reduction: that is owned by the
        // Reduce node + reduce dispatcher, not a localizing transformation.
        if (container_written) {
            bool intra_block_coop = has_cooperative_at(Level::Group) || has_cooperative_at(Level::Subgroup);
            bool owned_per_thread = false;
            for (const auto& a : private_axes()) {
                if (a.schedule().level() == Level::Group || a.schedule().level() == Level::Subgroup) {
                    owned_per_thread = true;
                    break;
                }
            }
            // Decline a genuine intra-block/warp reduction (owned by Reduce), or a
            // grid-cooperative write with no per-thread owner (a real cross-block
            // reduction needing atomics/grid sync). But a grid-only "cooperative"
            // write that a finer per-thread block axis already addresses is disjoint
            // per-block output — a private per-thread register tile (fall through).
            if (intra_block_coop || !owned_per_thread) {
                return std::nullopt;
            }
        } else {
            // A cooperative buffer lives in a device scope inside the kernel, below
            // the outermost loop.
            if (!inside_scratchpad_scope() || loop_is_outermost_) {
                return std::nullopt;
            }
            // Storage follows the finest cooperative level that owns a real buffer.
            // A read tile cooperative within a block lives in shared memory even when
            // it is also grid-cooperative: each block redundantly stages its own copy
            // (grid cooperation is replication, not a shared buffer). Only *pure* grid
            // cooperation needs a grid-wide global buffer.
            if (has_cooperative_at(Level::Group)) {
                return Space::Shared;
            }
            if (has_cooperative_at(Level::Device)) {
                return Space::Global;
            }
            // Warp-only cooperation is served by shuffles, not a staged buffer.
            return std::nullopt;
        }
    }
    // No cooperative axes: a thread-private / sequential buffer. But a host-level
    // loop that is itself scratchpad-scheduled or wraps a scratchpad kernel is not a
    // site for a private stack buffer.
    if (!inside_scratchpad_scope() && (loop_has_scratchpad_ || has_scratchpad_descendant_)) {
        return std::nullopt;
    }
    return Space::Register;
}

LocalityPlan Tile::placement(structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager)
    const {
    return LocalityPlan::analyze(loop, axes_, analysis_manager);
}

} // namespace tiles
} // namespace sdfg
