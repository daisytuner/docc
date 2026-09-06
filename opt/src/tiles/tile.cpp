#include "sdfg/tiles/tile.h"

#include <utility>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/tiles/tile_target_registry.h"

namespace sdfg {
namespace tiles {

Space default_space(Level level) {
    switch (level) {
        case Level::Device:
            return Space::Global; // device-wide cooperation (GPU grid / CPU threads): only global memory
        case Level::Group:
            return Space::Shared;
        case Level::Subgroup:
            return Space::Register;
    }
    return Space::Register;
}

AxisSchedule::AxisSchedule(
    Level level, Space space, bool has_scratchpad, unsigned spatial_axis, symbolic::Integer parallel_size, bool needs_sync
)
    : level_(level), space_(space), has_scratchpad_(has_scratchpad), spatial_axis_(spatial_axis),
      parallel_size_(std::move(parallel_size)), needs_sync_(needs_sync) {}

std::optional<AxisSchedule> AxisSchedule::classify(const structured_control_flow::ScheduleType& sched) {
    // The target that owns this schedule value supplies the classification.
    if (auto* target = TileTargetRegistry::instance().get(sched.value())) {
        return target->classify(sched);
    }
    // No registered target: a sequential loop does not shape storage; any other
    // parallel schedule cooperates device-wide on the host (global memory, no
    // scratchpad).
    if (sched.category() == structured_control_flow::ScheduleTypeCategory::None) {
        return std::nullopt;
    }
    return AxisSchedule(Level::Device, Space::Global, /*has_scratchpad=*/false);
}

std::optional<Level> AxisSchedule::classify_level(const structured_control_flow::ScheduleType& sched) {
    auto schedule = classify(sched);
    if (schedule && schedule->has_scratchpad()) {
        return schedule->level();
    }
    return std::nullopt;
}

bool AxisSchedule::drives_cooperative_copy(const structured_control_flow::ScheduleType& sched) {
    auto* target = TileTargetRegistry::instance().get(sched.value());
    return target && target->supports_cooperative_staging(sched) && classify_level(sched) == Level::Group;
}

TileAxis::TileAxis(
    symbolic::Symbol indvar, Role role, AxisSchedule schedule, symbolic::Expression init, symbolic::Integer stride
)
    : indvar_(std::move(indvar)), role_(role), schedule_(std::move(schedule)), init_(std::move(init)),
      stride_(std::move(stride)) {}

std::vector<TileAxis> TileAxis::
    enclosing(structured_control_flow::StructuredLoop& loop, const symbolic::MultiExpression& bases) {
    // An axis is cooperative when its indvar addresses no tile base (all
    // iterations share the same tile); otherwise it is per-iteration private.
    auto is_cooperative = [&](const symbolic::Symbol& indvar) {
        for (const auto& base : bases) {
            if (symbolic::uses(base, indvar)) {
                return false;
            }
        }
        return true;
    };
    std::vector<TileAxis> axes;
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        auto* sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(node);
        if (sloop == nullptr) {
            continue;
        }
        auto facts = AxisSchedule::classify(sloop->schedule_type());
        if (!facts) {
            continue; // sequential loop — does not shape storage
        }
        symbolic::Symbol indvar = sloop->indvar();
        Role role = is_cooperative(indvar) ? Role::Cooperative : Role::Private;
        symbolic::Integer stride = symbolic::integer(1);
        if (auto s = sloop->stride(); !s.is_null()) {
            stride = s;
        }
        axes.emplace_back(indvar, role, *facts, sloop->init(), stride);
    }
    return axes;
}

Tile::Tile(std::string container, Layout source, std::vector<TileAxis> axes, bool reads, bool writes)
    : container_(std::move(container)), source_(std::move(source)), axes_(std::move(axes)), reads_(reads),
      writes_(writes) {}

bool Tile::cooperative() const {
    for (const auto& x : axes_) {
        if (x.cooperative()) {
            return true;
        }
    }
    return false;
}

std::vector<TileAxis> Tile::cooperative_axes() const {
    std::vector<TileAxis> out;
    for (const auto& x : axes_) {
        if (x.cooperative()) {
            out.push_back(x);
        }
    }
    return out;
}

std::vector<TileAxis> Tile::private_axes() const {
    std::vector<TileAxis> out;
    for (const auto& x : axes_) {
        if (!x.cooperative()) {
            out.push_back(x);
        }
    }
    return out;
}

Space Tile::required_space() const {
    bool global = false;
    bool shared = false;
    for (const auto& x : axes_) {
        if (!x.cooperative()) {
            continue;
        }
        if (x.schedule().space() == Space::Global) {
            global = true;
        } else if (x.schedule().space() == Space::Shared) {
            shared = true;
        }
    }
    if (global) {
        return Space::Global;
    }
    if (shared) {
        return Space::Shared;
    }
    return Space::Register; // warp-cooperative (shuffle) or fully private
}

} // namespace tiles
} // namespace sdfg
