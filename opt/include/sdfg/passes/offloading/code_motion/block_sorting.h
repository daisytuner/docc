#pragma once

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace sdfg {
namespace passes {

class BlockSortingPass : public Pass {
protected:
    virtual bool is_libnode_side_effect_white_listed(data_flow::LibraryNode* libnode);

    virtual bool can_be_bubbled_up(structured_control_flow::Block& block);
    virtual bool can_be_bubbled_down(structured_control_flow::Block& block);

    virtual std::pair<int, std::string> get_prio_and_order(structured_control_flow::Block* block);

    bool is_reference_block(structured_control_flow::Block& block);

    bool is_libnode_block(structured_control_flow::Block& next_block);

public:
    virtual std::string name() override {
        return "BlockSorting";
    }

    bool bubble_up(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence,
        long long index
    );
    bool bubble_down(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence,
        long long index
    );

    bool sort(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence
    );

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
