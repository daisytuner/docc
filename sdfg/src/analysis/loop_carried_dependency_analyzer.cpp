#include "sdfg/analysis/loop_carried_dependency_analyzer.h"

namespace sdfg::analysis {

void NewLoopDependencyAnalysis::ScopeState::direct_write(DepAccess& access) {
    auto& space = this->direct_accesses_[access.container_];
    if (!space) {
        space = std::make_unique<GroupedAccesses>();
    }
    auto& grouped = *space;
    auto entries = grouped.de_writes.size();
    if (entries > 0) {
        if (entries > 1) {
            throw std::runtime_error("Multiple DE direct writes to " + access.container_ + "?");
        }

        auto* prev = grouped.de_writes.front();
        if (prev) {
            grouped.local_writes.push_back(prev);
        }
    }
    grouped.de_writes[0] = &access;
}

void NewLoopDependencyAnalysis::ScopeState::indirect_read(DepAccess& access) {}

void NewLoopDependencyAnalysis::ScopeState::indirect_write(DepAccess& access) {}

bool LoopDependencyInfo::available() const { return valid_; }

const std::unordered_map<std::string, LoopCarriedDependencyInfo>& LoopDependencyInfo::dependencies() const {
    return dependencies_;
}

bool LoopDependencyInfo::has_loop_carried() const {
    return false; // TODO
}

bool LoopDependencyInfo::has_loop_carried_raw() const {
    return false; // TODO
}

void NewLoopDependencyAnalysis::ScopeState::direct_read(DepAccess& access) {
    auto& space = this->direct_accesses_[access.container_];
    if (!space) {
        space = std::make_unique<GroupedAccesses>();
    }
    auto& grouped = *space;

    auto entries = grouped.de_writes.size();
    if (entries > 0) {
        if (entries > 1) {
            throw std::runtime_error("Multiple DE direct writes to " + access.container_ + "?");
        }
        // read of the most current value
    } else {
        // grouped.ue_reads.push_back()
    }
}

NewLoopDependencyAnalysis::ScopeState* NewLoopDependencyAnalysis::get_current_scope() {
    if (current_scope_stack_.empty()) {
        return nullptr;
    } else {
        return &current_scope_stack_.back();
    }
}

NewLoopDependencyAnalysis::NewLoopDependencyAnalysis(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis)
    : sdfg_(sdfg), analysis_(analysis), DataDependencyAnalyzer(sdfg) {
    detailed_assumptions_ = std::make_unique<AssumptionsAnalysis>(sdfg, true);
    detailed_assumptions_->run(analysis);
}

const LoopDependencyInfo& NewLoopDependencyAnalysis::get_analysis(StructuredLoop& loop) {
    auto it = loop_dependency_info_.find(loop.element_id());
    if (it == loop_dependency_info_.end()) {
        analyze_loop(loop);
    }
    return *loop_dependency_info_.at(loop.element_id());
}

void NewLoopDependencyAnalysis::analyze_entire_sdfg() {
    current_scope_stack_.clear();
    analysis_state_ = std::make_unique<AnalysisState>();
    dispatch(sdfg_.root());
}

void NewLoopDependencyAnalysis::analyze_loop(StructuredLoop& loop) {
    // clear state
    current_scope_stack_.clear();
    analysis_state_ = std::make_unique<AnalysisState>();
    dispatch(loop);
}

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

void NewLoopDependencyAnalysis::direct_read(DepAccess& access) {
    if (auto* scope = get_current_scope()) {
        scope->direct_read(access);
    }
}

void NewLoopDependencyAnalysis::direct_write(DepAccess& access) {
    if (auto* scope = get_current_scope()) {
        scope->direct_write(access);
    }
}

void NewLoopDependencyAnalysis::indirect_read(DepAccess& access) {
    if (auto* scope = get_current_scope()) {
        scope->indirect_read(access);
    }
}

void NewLoopDependencyAnalysis::indirect_write(DepAccess& access) {
    auto* scope = get_current_scope();
    if (scope) {
        scope->indirect_write(access);
    }
}

void NewLoopDependencyAnalysis::aliasing_source(const std::string& container) {
    if (auto* scope = get_current_scope()) {
        DEBUG_PRINTLN(
            "Aliasing source detected for container: " << container << ", inside scope #" << scope->scope_->element_id()
        );
        scope->not_understood();
    }
}


} // namespace sdfg::analysis
