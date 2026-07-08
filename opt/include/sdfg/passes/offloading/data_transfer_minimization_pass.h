#pragma once

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/control_flow_analysis.h"
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
        const analysis::DataTransferEliminationAnalysis& transfer_analysis,
        analysis::ControlFlowAnalysis& cf_analysis,
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


} // namespace passes
} // namespace sdfg
