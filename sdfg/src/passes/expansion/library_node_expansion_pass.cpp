#include "sdfg/passes/expansion/library_node_expansion_pass.h"

#include "lib_node_expansion_context.h"
#include "sdfg/data_flow/library_nodes/math/math.h"

#include <memory>

namespace sdfg {
namespace passes {

LibNodeExpander::ExpandOutcome MathNodeExpander::handle_expand(ExpandContext& context, Block& block, math::MathNode& node)
    const {
    return node.expand(context, block);
}

LibNodeExpansionVisitor::NodeOutcome LibNodeExpansionVisitor::
    try_expand(Sequence& parent, size_t child_idx, Block& block, sdfg::data_flow::LibraryNode& node) {
    auto expander = expander_.for_lib_node(node);

    if (expander) {
        expansion::LibNodeExpansionContext ctx(this->builder_, parent, child_idx, block, node);

        auto outcome = expander->handle_expand(ctx, block, node);

        if (ctx.expanded()) {
            ctx.cleanup();
            return {true, ctx.dropped_block()};
        }
    }

    return {};
}

LibNodeExpansionVisitor::LibNodeExpansionVisitor(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, Expanders& expander
)
    : visitor::ActualStructuredSDFGVisitor(), builder_(builder), analysis_manager_(analysis_manager),
      expander_(expander) {}

bool LibNodeExpansionVisitor::visit(sdfg::structured_control_flow::Sequence& seq) {
    bool may_contain_libnodes = true;
    size_t i = 0;

    // expansion can remove and replace entire blocks, so need smarter sequence visiting that can keep iterating even
    // when the children change expansion can only remove the current node and add 1 or more children in its stead, so
    // idx will never decrease but we do need the info what was changed to avoid invalidated parts

    do {
        size_t total_children = seq.size();
        if (total_children == 0) {
            may_contain_libnodes = false;
        } else if (i >= total_children) {
            return true;
        }

        for (; i < total_children; ++i) {
            auto& child = seq.at(i).first;
            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&child)) {
                auto outcome = handle_block(seq, i, *block);

                if (outcome.block_removed) { // recheck with same i, its now a new block
                    if (outcome.skip_count) {
                        i += outcome.skip_count;
                    }
                    break;
                } else if (outcome.skip_count) {
                    i += outcome.skip_count;
                    // +1 from for loop
                }
            } else {
                dispatch(child);
            }
        }
    } while (may_contain_libnodes);

    return true;
}

LibNodeExpansionVisitor::BlockOutcome LibNodeExpansionVisitor::
    handle_block(structured_control_flow::Sequence& parent, size_t child_idx, structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();

    ElementId last_element_id = 0;

    bool may_contain_lib_nodes = true;
    bool handled_any = false;

    do {
        // expansion may change the contents of this block or even remove it.
        // to ensure stable order, order by element_id
        // track the last handled element_id, because changes are not allowed  to affect other libnodes.
        // So if the current block gets invalidated (but not removed) we can restart iterating above the last processed
        // element_id
        auto libnodes = dataflow.nodes() |
                        std::views::transform([](auto& n) { return dynamic_cast<const data_flow::LibraryNode*>(&n); }) |
                        std::views::filter([](auto* n) { return n != nullptr; }) |
                        std::views::filter([last_element_id](auto* n) {
                            return n->implementation_type() == data_flow::ImplementationType_NONE &&
                                   n->element_id() > last_element_id;
                        });
        std::vector<const data_flow::LibraryNode*> sorted_nodes(libnodes.begin(), libnodes.end());
        std::ranges::sort(sorted_nodes, std::less<>{}, [](const auto* n) { return n->element_id(); });

        may_contain_lib_nodes = !libnodes.empty();

        bool block_changed = false;
        for (auto* library_node : sorted_nodes) {
            last_element_id = library_node->element_id();
            auto outcome = try_expand(parent, child_idx, block, *const_cast<data_flow::LibraryNode*>(library_node));
            handled_any |= outcome.expanded;

            if (outcome.block_removed) {
                this->expanded_any_ = true;
                return static_cast<BlockOutcome>(outcome);
            } else if (outcome.expanded) {
                // libnodes inside this block have become invalid
                block_changed = true;
                break;
            }
        }
        if (block_changed) { // here as a reminder, that we must not do anything else in this iteration, as the
                             // underlying nodes might be invalid now
            continue;
        }
    } while (may_contain_lib_nodes);

    if (handled_any) {
        this->expanded_any_ = true;
    }

    return {};
}

bool LibraryNodeExpansionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    LibNodeExpansionVisitor v(builder, analysis_manager, *expander_.get());

    v.dispatch(builder.subject().root());

    return v.expanded_any_;
}

LibNodeExpansionVisitor::NodeOutcome expansion::expand_single_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    data_flow::LibraryNode& node,
    const LibNodeExpansionVisitor::Expanders& expander
) {
    auto* exp = expander.for_lib_node(node);

    if (!exp) {
        return {};
    }

    auto& seq = *dynamic_cast<Sequence*>(block.get_parent());

    auto idx = seq.index(block);

    LibNodeExpansionContext ctx(builder, seq, idx, block, node);

    auto outcome = exp->handle_expand(ctx, block, node);

    if (ctx.expanded()) {
        ctx.cleanup();
        return {true, ctx.dropped_block()};
    }

    return {};
}

LibNodeExpansionVisitor::NodeOutcome expansion::
    expand_single_math_node(builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node) {
    return expand_single_node(builder, block, node, MathNodeExpander());
}

} // namespace passes
} // namespace sdfg
