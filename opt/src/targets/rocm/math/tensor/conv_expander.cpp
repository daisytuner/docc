#include "sdfg/targets/rocm/math/tensor/conv_expander.h"

#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/cuda/math/tensor/conv_expander.h"

namespace sdfg {
namespace offloading {

passes::LibNodeExpander::ExpandOutcome RocmConvExpander::handle_expand(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::ConvNode& node
) const {
    // for now we reuse the cuda impl until we find sth. where they need to diverge
    return CudaConvExpander::handle_expand_im2row(context, block, node);
}

} // namespace offloading
} // namespace sdfg
