/**
 * @file expansion_pass.h
 * @brief Library node expansion pass
 *
 * This file defines a peephole optimizer that will find all LibraryNodes with ImplementationType_NONE in the graph
 * and attempt to expand all of them into more baser operations.
 * It internally uses a list of LibNodeExpanders, so there could be multiple alternative variants to expand a given
 * node, but for now it simply redirects to MathNode.expand() to stay more compatible with existing code.
 *
 * The pass tries to isolate the contents of the expansion from the surrounding SDFG as best as possible,
 * handling finding access nodes or moving edges in a generic way. This makes expansions simpler and allows us to
 * upgrade the passes' handling in the future, for example to cut Dataflow into multiple parts to allow for expansion.
 *
 * To achieve this, every Expander is given a context and the block to do preliminary checks, whether it can /
 * wants to expand this node with its properties. Access to src and dest edges may be neccessary,
 * as some LibraryNodes currently infer their data types from other nodes. Access to src and dest nodes however should
 * be avoided, as the pass may change or remove them.
 *
 * In case, an Expander wants to proceed with the expansion, it needs to call one of the available methods on the
 * ExpansionContext:
 *  * replacement_requires_access_nodes()
 *    * The classic way, where every input and output edge is going to be replaced with AccessNodes.
 *    This way, the inputs might be accessed inside newly created loops etc.
 *  * replace_dataflow()
 *    * A simpler way, where the replacement will happen inside the existing DataFlowGraph. In this case,
 *    no access nodes will be touched. The Expander will create its new nodes and define where which original input and
 * output edge will go The generic expansion logic will then check if this is possible or the graph can be changed to
 * make this possible (future improvements here) It might return nullptr, if it does not know how to isolate the
 * LibraryNode Otherwise, if the replacement was started, it must be finished successfully to not leave the SDFG in a
 * possibly broken state.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class LibraryNodeExpansionPass;

class MathNodeExpander : public TypedLibNodeExpander<math::MathNode> {
public:
    LibNodeExpander::ExpandOutcome handle_expand(ExpandContext& context, Block& block, math::MathNode& node)
        const override;
};

class LibNodeExpansionVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend LibraryNodeExpansionPass;

public:
    using Expanders = LibNodeExpander;

    struct BlockOutcome {
        bool block_removed;
        int skip_count;

        BlockOutcome(bool block_removed = false, int skip_count = 0)
            : block_removed(block_removed), skip_count(skip_count) {}
    };

    struct NodeOutcome : public BlockOutcome {
        bool expanded;

        NodeOutcome(bool expanded = false, bool block_removed = false, int skip_count = 0)
            : BlockOutcome(block_removed, skip_count), expanded(expanded) {}
    };

private:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;
    bool expanded_any_ = false;
    Expanders& expander_;


    NodeOutcome try_expand(Sequence& parent, size_t child_idx, Block& block, sdfg::data_flow::LibraryNode& node);

public:
    /**
     * @brief Construct the expansion visitor
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying properties
     */
    LibNodeExpansionVisitor(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, Expanders& expander
    );

    bool visit(sdfg::structured_control_flow::Sequence& seq) override;

    BlockOutcome handle_block(
        structured_control_flow::Sequence& parent, size_t child_idx, sdfg::structured_control_flow::Block& block
    );
};

/**
 * @class LibraryNodeExpansionPass
 * @brief Looks for and expands library nodes in a single pass, potentially recursively
 */
class LibraryNodeExpansionPass : public Pass {
    std::shared_ptr<LibNodeExpansionVisitor::Expanders> expander_;

public:
    LibraryNodeExpansionPass() : expander_(std::make_shared<MathNodeExpander>()) {}
    LibraryNodeExpansionPass(std::shared_ptr<LibNodeExpansionVisitor::Expanders> expander)
        : expander_(std::move(expander)) {}

    std::string name() override { return "LibraryNodeExpansionPass"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

namespace expansion {

LibNodeExpansionVisitor::NodeOutcome expand_single_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    data_flow::LibraryNode& node,
    const LibNodeExpansionVisitor::Expanders& expanders
);

LibNodeExpansionVisitor::NodeOutcome
expand_single_math_node(builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node);

} // namespace expansion

} // namespace passes
} // namespace sdfg
