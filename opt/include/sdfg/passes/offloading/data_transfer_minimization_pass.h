#pragma once

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_transfer_elimination_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class DataTransferMinimizationPass : public Pass {
public:
    DataTransferMinimizationPass();

    std::string name() override { return "DataTransferMinimization"; };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    static bool available(analysis::AnalysisManager& AM) { return true; }

protected:
    bool eliminate_transfer_pair(
        builder::StructuredSDFGBuilder& builder,
        analysis::OffloadHolder& copy_out,
        analysis::OffloadHolder& copy_in,
        bool remove_d2h = false
    );
    bool eliminate_malloc_first_transfer(
        builder::StructuredSDFGBuilder& builder,
        analysis::OffloadHolder& malloc_holder,
        analysis::OffloadHolder& copy_in
    );
    bool eliminate_redundant_d2h(
        builder::StructuredSDFGBuilder& builder, analysis::OffloadHolder& h2d, analysis::OffloadHolder& d2h
    );
};

class DataTransferMinimizationLegacy : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    virtual std::pair<data_flow::AccessNode*, data_flow::AccessNode*>
    get_src_and_dst(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node);

protected:
    data_flow::AccessNode* get_in_access(data_flow::CodeNode* node, const std::string& dst_conn);
    data_flow::AccessNode* get_out_access(data_flow::CodeNode* node, const std::string& src_conn);

    bool check_container_dependency(
        structured_control_flow::Block* copy_out_block,
        const std::string& copy_out_container,
        structured_control_flow::Block* copy_in_block,
        const std::string& copy_in_container
    );

public:
    DataTransferMinimizationLegacy(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "DataTransferMinimization"; }

    virtual bool visit() override;

    virtual bool accept(structured_control_flow::Sequence& sequence) override;
};

typedef VisitorPass<DataTransferMinimizationLegacy> DataTransferMinimizationLegacyPass;

} // namespace passes
} // namespace sdfg
