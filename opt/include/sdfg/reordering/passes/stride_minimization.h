#pragma once

#include <sdfg/passes/pass.h>

namespace sdfg {
namespace reordering {

class StrideMinimization : public passes::Pass {
private:
    std::pair<bool, std::vector<std::string>> can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::ControlFlowNode*>& nested_loops
    );

    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::ControlFlowNode*>& nested_loops,
        std::vector<std::string> target_permutation
    );

public:
    StrideMinimization();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    static bool is_admissible(
        std::vector<std::string>& current,
        std::vector<std::string>& target,
        std::unordered_set<std::string>& allowed_swaps
    );

    static std::unordered_set<std::string> allowed_swaps(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::ControlFlowNode*>& nested_loops
    );
};

} // namespace reordering
} // namespace sdfg
