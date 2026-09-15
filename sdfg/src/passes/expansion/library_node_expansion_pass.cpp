#include "sdfg/passes/expansion/library_node_expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/passes/expansion/lib_node_expansion_context.h"

#include <memory>

namespace sdfg {
namespace passes {

LibNodeExpander::ExpandOutcome MathNodeExpander::handle_expand(ExpandContext& context, Block& block, math::MathNode& node)
    const {
    return node.expand(context, block);
}

expansion::NodeOutcome expansion::expand_single_node(
    builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node, const LibNodeExpander& expander
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

expansion::NodeOutcome expansion::
    expand_single_math_node(builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node) {
    return expand_single_node(builder, block, node, MathNodeExpander());
}

} // namespace passes
} // namespace sdfg
