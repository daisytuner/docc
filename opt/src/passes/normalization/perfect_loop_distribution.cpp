#include "sdfg/passes/normalization/perfect_loop_distribution.h"

#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>
#include <unordered_set>
#include "sdfg/analysis/loop_analysis.h"

namespace sdfg {
namespace passes {
namespace normalization {

bool PerfectLoopDistributionPass::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    if (loop.root().size() == 1) {
        return false;
    }

    bool has_subloop = false;
    for (size_t i = 0; i < loop.root().size(); i++) {
        // skip blocks
        if (dynamic_cast<structured_control_flow::Block*>(&loop.root().at(i).first)) {
            continue;
        }
        if (dynamic_cast<structured_control_flow::StructuredLoop*>(&loop.root().at(i).first)) {
            has_subloop = true;
            break;
        }
        // if not a block or a loop, then we can't apply the transformation
        return false;
    }
    if (!has_subloop) {
        return false;
    }

    transformations::LoopDistribute loop_distribute(loop);
    if (!loop_distribute.can_be_applied(builder, analysis_manager)) {
        return false;
    }

    return true;
};

void PerfectLoopDistributionPass::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    transformations::LoopDistribute loop_distribute(loop);
    loop_distribute.apply(builder, analysis_manager);
};

PerfectLoopDistributionPass::PerfectLoopDistributionPass()
    : passes::Pass() {

      };


bool PerfectLoopDistributionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& outermost_loops = loop_analysis.outermost_loops();
    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> loop_tree_paths;

    for (auto& loop : outermost_loops) {
        loop_tree_paths.splice(loop_tree_paths.end(), loop_analysis.loop_tree_paths(loop));
    }

    if (loop_tree_paths.empty()) {
        // No loops to process, so we can exit early
        return false;
    }

    // sort paths by depth (deepest first)
    loop_tree_paths.sort([](const std::vector<sdfg::structured_control_flow::ControlFlowNode*>& a,
                            const std::vector<sdfg::structured_control_flow::ControlFlowNode*>& b) {
        return a.size() > b.size();
    });

    bool changed = false;
    auto max_depth = loop_tree_paths.front().size();
    for (size_t depth = max_depth; depth > 0; depth--) {
        // get individual loops at this depth
        std::unordered_set<structured_control_flow::StructuredLoop*> loops_at_depth;
        for (auto& path : loop_tree_paths) {
            if (path.size() < depth) {
                // We can break here because paths are sorted by depth, so all subsequent paths will be shallower
                break;
            }
            auto* node = path[depth - 1];
            auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(node);
            if (loop) {
                loops_at_depth.insert(loop);
            }
        }

        // find applicable loops at this depth
        std::unordered_set<structured_control_flow::StructuredLoop*> applicable_loops;
        for (auto* loop : loops_at_depth) {
            if (this->can_be_applied(builder, analysis_manager, *loop)) {
                applicable_loops.insert(loop);
            }
        }

        // apply transformation to applicable loops
        for (auto* loop : applicable_loops) {
            this->apply(builder, analysis_manager, *loop);
            changed = true;
        }
    }
    return changed;
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
