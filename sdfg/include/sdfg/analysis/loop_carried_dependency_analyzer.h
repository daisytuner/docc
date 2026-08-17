#pragma once

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/data_dependency_analyzer.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg::analysis {

struct LoopDependencyInfo {};

typedef std::string RegId;

class LoopDepDataDepShim;

struct GroupedAccesses {
    DepAccess* owning_write = nullptr;
    std::vector<DepAccess*> ue_reads;
    std::vector<DepAccess*> local_writes;
    std::vector<DepAccess*> de_writes;
};

class NewLoopDependencyAnalysis : BaseUserVisitor {
    friend LoopDepDataDepShim;

    StructuredSDFG& sdfg_;
    AnalysisManager& analysis_;
    std::unique_ptr<AssumptionsAnalysis> detailed_assumptions_;

    class ScopeState {
        friend LoopDepDataDepShim;

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

    std::list<ScopeState> current_scope_stack_;
    std::unique_ptr<AnalysisState> analysis_state_;

    std::unordered_map<ElementId, std::unique_ptr<LoopDependencyInfo>> loop_dependency_info_;

public:
    NewLoopDependencyAnalysis(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis);

    void analyze_entire_sdfg();

    void analyze_loop(StructuredLoop& loop);

    bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) override;

    bool visit(sdfg::structured_control_flow::While& node) override;

    bool visit(sdfg::structured_control_flow::IfElse& node) override;

protected:
    void capture_de_on_exit(AnalysisState& state, ScopeState& loop_state);
};

class LoopDepDataDepShim : public DataDependencyAnalyzer {
    NewLoopDependencyAnalysis& analysis_;

public:
    explicit LoopDepDataDepShim(NewLoopDependencyAnalysis& analysis, StructuredSDFG& sdfg)
        : analysis_(analysis), DataDependencyAnalyzer(sdfg) {}

protected:
    void direct_read(DepAccess& access) override;
    void direct_write(DepAccess& access) override;
    void indirect_read(DepAccess& access) override;
    void indirect_write(DepAccess& access) override;
    void aliasing_source(const std::string& container) override;
};

} // namespace sdfg::analysis
