#pragma once

#include <cstddef>
#include <set>
#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace passes {

class ReadonlyTransferHoistingPass : public Pass {
private:
    std::pair<structured_control_flow::Sequence*, size_t> get_first_location(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const std::string& container
    );

    bool move_readonly_transfer(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const std::string& container,
        structured_control_flow::Sequence* sequence,
        size_t index,
        std::set<size_t>& visisted
    );

    long long find_matching_free_block(
        structured_control_flow::Sequence* parent,
        structured_control_flow::Block* block,
        offloading::DataOffloadingNode* offloading_node
    );

    std::pair<structured_control_flow::Sequence*, size_t> get_safe_location(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence* sequence,
        size_t index,
        structured_control_flow::Block* block,
        offloading::DataOffloadingNode* offloading_node
    );

    std::pair<structured_control_flow::Sequence*, size_t> correct_location(
        analysis::ScopeAnalysis& scope_analysis,
        structured_control_flow::Sequence* sequence,
        size_t index,
        const std::vector<std::pair<structured_control_flow::Sequence*, size_t>>& parents
    );
    std::vector<std::pair<structured_control_flow::Sequence*, size_t>>
    get_parents(analysis::ScopeAnalysis& scope_analysis, structured_control_flow::Block* block);

public:
    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
