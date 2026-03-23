#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace passes {

/**
 * Currently only looks at the first and lost block in the sequence of a if/else or Map node and thus provides very
 * limited hoisting by itself
 */
class BlockHoisting : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool is_invariant_move(
        structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, bool no_loop_carried_dependencies = true
    );
    bool is_invariant_view(
        structured_control_flow::Sequence& body,
        data_flow::DataFlowGraph& dfg,
        symbolic::Symbol indvar = SymEngine::null,
        bool no_loop_carried_dependencies = true
    );
    bool is_invariant_libnode(
        structured_control_flow::Sequence& body,
        data_flow::DataFlowGraph& dfg,
        symbolic::Symbol indvar = SymEngine::null,
        bool no_loop_carried_dependencies = true
    );

    bool equal_moves(structured_control_flow::Block& block1, structured_control_flow::Block& block2);
    bool equal_views(structured_control_flow::Block& block1, structured_control_flow::Block& block2);
    bool equal_libnodes(data_flow::LibraryNode* libnode1, data_flow::LibraryNode* libnode2);
    bool equal_lib_blocks(structured_control_flow::Block& block1, structured_control_flow::Block& block2);

    bool map_invariant_front(structured_control_flow::Sequence& parent, structured_control_flow::Map& map_stmt);
    bool map_invariant_back(structured_control_flow::Sequence& parent, structured_control_flow::Map& map_stmt);
    bool map_invariant_move(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );
    bool map_invariant_view(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );
    bool map_invariant_libnode_front(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );
    bool map_invariant_libnode_back(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );

    bool if_else_invariant_front(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else);
    bool if_else_invariant_back(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else);
    void if_else_extract_invariant_front(
        structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
    );
    void
    if_else_extract_invariant_back(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else);

    bool equal_offloading_nodes(
        structured_control_flow::Block& block1,
        offloading::DataOffloadingNode* offloading_node1,
        structured_control_flow::Block& block2,
        offloading::DataOffloadingNode* offloading_node2
    );

protected:
    virtual bool is_libnode_allowed(
        structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, data_flow::LibraryNode* libnode
    );

    virtual bool equal_libnodes(structured_control_flow::Block& block1, structured_control_flow::Block& block2);

    virtual void if_else_extract_invariant_libnode_front(
        structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
    );
    virtual void if_else_extract_invariant_libnode_back(
        structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
    );

public:
    BlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "BlockHoisting"; }

    virtual bool accept(structured_control_flow::Map& map_stmt) override;

    virtual bool accept(structured_control_flow::IfElse& if_else) override;
};

typedef VisitorPass<BlockHoisting> BlockHoistingPass;

} // namespace passes
} // namespace sdfg
