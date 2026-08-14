#include "sdfg/analysis/loop_carried_dependency_analyzer.h"

namespace sdfg::analysis {

void NewLoopDependencyAnalysis::ScopeState::direct_write(DepAccess& access) {
    auto& list = de_writes[access.container_];
    auto entries = list.size();
    if (entries > 0) {
        if (entries > 1) {
            // there are indirect accesses, whose base has likely become invalid and we need to be uncertain about
            // matching them up with new ones
        }
        auto* prev = list.front() if (prev) { local_writes[access.container_].push_back(prev); }
    }
    list.push_back(&access);
}

NewLoopDependencyAnalysis::NewLoopDependencyAnalysis(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis)
    : sdfg_(sdfg), analysis_(analysis) {
    detailed_assumptions_ = std::make_unique<AssumptionsAnalysis>(sdfg, true);
    detailed_assumptions_->run(analysis);
}

void NewLoopDependencyAnalysis::analyze_loop(StructuredLoop& loop) { dispatch(loop); }

bool NewLoopDependencyAnalysis::handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) {
    current_scope_stack_.emplace_back(loop);
    auto& elem = current_scope_stack_.back();

    if (!loop.is_monotonic()) {
        elem.not_understood();
    } else {
    }

    auto outcome = BaseUserVisitor::handleStructuredLoop(loop);

    if (!elem.is_not_understood()) {
        // commit result to loop_dependency_info_
        capture_de_on_exit(*analysis_state_, elem);
    }

    current_scope_stack_.pop_back();

    return outcome;
}

void NewLoopDependencyAnalysis::capture_de_on_exit(AnalysisState& analysis_state, ScopeState& loop_state) {}

void LoopDepDataDepShim::direct_read(DepAccess& access) { analysis_.current_scope_stack_.back().direct_read(access); }

void LoopDepDataDepShim::direct_write(DepAccess& access) { analysis_.current_scope_stack_.back().direct_write(access); }

void LoopDepDataDepShim::indirect_read(DepAccess& access) {
    analysis_.current_scope_stack_.back().indirect_read(access);
}

void LoopDepDataDepShim::indirect_write(DepAccess& access) {
    analysis_.current_scope_stack_.back().indirect_write(access);
}

void LoopDepDataDepShim::aliasing_source(const std::string& container) {
    auto& current_scope = analysis_.current_scope_stack_.back();
    DEBUG_PRINTLN(
        "Aliasing source detected for container: " << container << ", inside scope #"
                                                   << current_scope.scope_->element_id()
    );
    current_scope.not_understood();
}


} // namespace sdfg::analysis
