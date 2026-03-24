#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/offloading/code_motion/block_hoisting.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace passes {

class ExtendedBlockHoisting : public BlockHoisting {
private:
    bool equal_offloading_nodes(
        structured_control_flow::Block& block1,
        offloading::DataOffloadingNode* offloading_node1,
        structured_control_flow::Block& block2,
        offloading::DataOffloadingNode* offloading_node2
    );
    data_flow::Memlet*
    get_offloading_node_iedge(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node);

protected:
    virtual bool is_libnode_allowed(
        structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, data_flow::LibraryNode* libnode
    ) override;

    virtual bool equal_libnodes(structured_control_flow::Block& block1, structured_control_flow::Block& block2) override;

    virtual void if_else_extract_invariant_libnode_front(
        structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
    ) override;
    virtual void if_else_extract_invariant_libnode_back(
        structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
    ) override;

public:
    ExtendedBlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "ExtendedBlockHoisting"; }
};

typedef VisitorPass<ExtendedBlockHoisting> ExtendedBlockHoistingPass;

} // namespace passes
} // namespace sdfg
