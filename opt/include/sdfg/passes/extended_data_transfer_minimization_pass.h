#pragma once

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class ExtendedDataTransferMinimization : public DataTransferMinimization {
private:
    virtual std::pair<data_flow::AccessNode*, data_flow::AccessNode*>
    get_src_and_dst(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node);

public:
    ExtendedDataTransferMinimization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "ExtendedDataTransferMinimization"; }
};

typedef VisitorPass<ExtendedDataTransferMinimization> ExtendedDataTransferMinimizationPass;

} // namespace passes
} // namespace sdfg
