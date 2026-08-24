#pragma once

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/data_dependency_analyzer.h"
#include "sdfg/analysis/loop_carried_dependency_info.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg::analysis {

class LoopDependencyInfo {
    bool valid_;
    std::unordered_map<std::string, LoopCarriedDependencyInfo> dependencies_;

public:
    bool available() const;
    const std::unordered_map<std::string, LoopCarriedDependencyInfo>& dependencies() const;
    bool has_loop_carried() const;
    bool has_loop_carried_raw() const;

    // reductions are a 2nd stage that can be computed on demand. If we need to cache that, we can think about caching
    // it in another layer
};

typedef std::string RegId;

struct GroupedAccesses {
    DepAccess* owning_write = nullptr;
    std::vector<DepAccess*> ue_reads;
    std::vector<DepAccess*> local_writes;
    std::vector<DepAccess*> de_writes;
};

class NewLoopDependencyAnalysis : BaseUserVisitor, DataDependencyAnalyzer {
    StructuredSDFG& sdfg_;
    AnalysisManager& analysis_;
    std::unique_ptr<AssumptionsAnalysis> detailed_assumptions_;

    class ScopeState {
        friend NewLoopDependencyAnalysis;

        ControlFlowNode* scope_;
        StructuredLoop* loop_;
        bool not_understood_ = false;

        std::unordered_map<RegId, std::unique_ptr<GroupedAccesses>> indirect_areas_;
        std::unordered_map<RegId, std::unique_ptr<GroupedAccesses>> direct_accesses_;

    public:
        ScopeState(StructuredLoop& loop) : scope_(&loop), loop_(&loop) {}
        ScopeState(ControlFlowNode& scope) : scope_(&scope), loop_(nullptr) {}

        void not_understood() { not_understood_ = true; }
        bool is_not_understood() const { return not_understood_; }

        void direct_read(DepAccess& access);
        void direct_write(DepAccess& access);
        void indirect_read(DepAccess& access);
        void indirect_write(DepAccess& access);
    };

    struct AnalysisState {};

    /// the current stack of nested scopes that need to traversed.
    /// If we only analyze a single loop, that loop may the outermost element on this stack,
    /// regardless if it has further parents, if we pop the parent, we are done with the current analysis
    std::list<ScopeState> current_scope_stack_;
    /// The live state we use as we traverse over the SDFG
    std::unique_ptr<AnalysisState> analysis_state_;

    std::unordered_map<ElementId, std::unique_ptr<LoopDependencyInfo>> loop_dependency_info_;

    ScopeState* get_current_scope();

protected:
    void direct_read(DepAccess& access) override;
    void direct_write(DepAccess& access) override;
    void indirect_read(DepAccess& access) override;
    void indirect_write(DepAccess& access) override;
    void aliasing_source(const std::string& container) override;

public:
    NewLoopDependencyAnalysis(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis);

    const LoopDependencyInfo& get_analysis(StructuredLoop& loop);

    void analyze_entire_sdfg();

    void analyze_loop(StructuredLoop& loop);

    bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) override;

    bool visit(sdfg::structured_control_flow::While& node) override;

    bool visit(sdfg::structured_control_flow::IfElse& node) override;

protected:
    void capture_de_on_exit(AnalysisState& state, ScopeState& loop_state);
};

} // namespace sdfg::analysis
