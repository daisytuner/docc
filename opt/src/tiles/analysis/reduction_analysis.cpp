#include "sdfg/tiles/analysis/reduction_analysis.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/tiles/tile.h"

namespace sdfg {
namespace tiles {

bool is_reduction_accumulator(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto owns = [&](structured_control_flow::ControlFlowNode* node) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (!reduce) {
            return false;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
        return false;
    };
    if (owns(&loop)) {
        return true;
    }
    for (auto* node : loop_analysis.ancestors(&loop)) {
        if (owns(node)) {
            return true;
        }
    }
    for (auto* node : loop_analysis.descendants(&loop)) {
        if (owns(node)) {
            return true;
        }
    }
    return false;
}

bool collect_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::Reduce*>& out
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto owns = [&](structured_control_flow::ControlFlowNode* node) -> structured_control_flow::Reduce* {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (!reduce) {
            return nullptr;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return reduce;
            }
        }
        return nullptr;
    };

    // A grid-parallel (e.g. split-K Z_GRID) offloaded reduce merges per-block
    // partials across the grid via an atomic writeback rather than in-place.
    auto is_grid_parallel_reduce = [&](structured_control_flow::Reduce* reduce) -> bool {
        return AxisSchedule::classify_level(reduce->schedule_type()) == Level::Device;
    };

    // An ancestor Reduce accumulates across iterations *outside* the localized
    // scope. Whether a buffer at loop is still legal depends on how its
    // per-outer-iteration writeback composes:
    //   - grid-parallel (GPU) ancestor: per-block partials merge via an atomic
    //     writeback; localization privatizes the partial and owns the merge (not
    //     retargeted here).
    //   - sequential (non-GPU) ancestor: outer iterations are barrier-separated, so
    //     a read-modify-write copy-in/out around loop carries the accumulation
    //     through the global container each iteration (classical BLIS pc loop). A
    //     racy cooperative-CPU writeback is rejected downstream by derive_storage.
    //   - any other (GPU block/warp cooperative) ancestor is combined by the reduce
    //     dispatcher and cannot be localized here.
    for (auto* node : loop_analysis.ancestors(&loop)) {
        if (auto* reduce = owns(node)) {
            if (is_grid_parallel_reduce(reduce)) {
                continue;
            }
            if (!AxisSchedule::classify_level(reduce->schedule_type()).has_value()) {
                continue;
            }
            return false;
        }
    }

    // loop itself or a descendant Reduce: privatizable only when the reduction is
    // combined sequentially / per-thread. A GPU-offloaded Reduce is combined across
    // threads by the reduce dispatcher, which owns the accumulator staging.
    auto consider = [&](structured_control_flow::Reduce* reduce) -> bool {
        if (AxisSchedule::classify_level(reduce->schedule_type()).has_value()) {
            return false;
        }
        out.push_back(reduce);
        return true;
    };
    if (auto* reduce = owns(&loop)) {
        if (!consider(reduce)) {
            return false;
        }
    }
    for (auto* node : loop_analysis.descendants(&loop)) {
        if (auto* reduce = owns(node)) {
            if (!consider(reduce)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<structured_control_flow::Reduce*> collect_grid_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    std::vector<structured_control_flow::Reduce*> out;
    for (auto* node : loop_analysis.ancestors(&loop)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (reduce == nullptr) {
            continue;
        }
        if (AxisSchedule::classify_level(reduce->schedule_type()) != Level::Device) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                out.push_back(reduce);
                break;
            }
        }
    }
    return out;
}

} // namespace tiles
} // namespace sdfg
