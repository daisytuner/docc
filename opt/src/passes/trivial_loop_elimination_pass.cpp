#include "sdfg/passes/trivial_loop_elimination_pass.h"

#include <deque>

#include "sdfg/transformations/trivial_loop_elimination.h"

namespace sdfg {
namespace passes {

TrivialLoopEliminationPass::TrivialLoopEliminationPass(std::vector<structured_control_flow::Map*>& maps) : maps_(maps) {
}

bool TrivialLoopEliminationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    std::vector<structured_control_flow::Map*> result;
    std::deque<structured_control_flow::Map*> worklist(maps_.begin(), maps_.end());
    while (!worklist.empty()) {
        auto* map = worklist.front();
        worklist.pop_front();

        transformations::TrivialLoopElimination transformation(*map);
        if (!transformation.can_be_applied(builder, analysis_manager)) {
            result.push_back(map);
            continue;
        }

        // Maps directly in the body take the place of the eliminated map
        std::vector<structured_control_flow::Map*> promoted;
        auto& body = map->root();
        for (size_t i = 0; i < body.size(); i++) {
            if (auto* child = dyn_cast<structured_control_flow::Map*>(&body.at(i))) {
                promoted.push_back(child);
            }
        }

        transformation.apply(builder, analysis_manager);
        applied = true;

        worklist.insert(worklist.begin(), promoted.begin(), promoted.end());
    }

    maps_ = std::move(result);
    return applied;
}

} // namespace passes
} // namespace sdfg
