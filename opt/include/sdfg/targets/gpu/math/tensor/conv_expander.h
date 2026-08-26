#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg {
namespace offloading {

class GPUConvExpander : public passes::CodeLibNodeExpander<math::tensor::ConvNode> {
public:
    GPUConvExpander() : passes::CodeLibNodeExpander<math::tensor::ConvNode>(math::tensor::LibraryNodeType_Conv) {};

    virtual passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context,
        structured_control_flow::Block& block,
        math::tensor::ConvNode& node
    ) const override;

    static passes::LibNodeExpander::ExpandOutcome handle_expand_im2row(
        passes::LibNodeExpander::ExpandContext& context,
        structured_control_flow::Block& block,
        math::tensor::ConvNode& node
    );
};
} // namespace offloading
} // namespace sdfg
