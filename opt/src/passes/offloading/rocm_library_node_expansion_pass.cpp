#include "sdfg/passes/offloading/rocm_library_node_expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/tensor/concat_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/targets/rocm/math/tensor/batched_matmul_expander.h"
#include "sdfg/targets/rocm/math/tensor/concat_expander.h"
#include "sdfg/targets/rocm/math/tensor/conv_expander.h"

namespace sdfg {
namespace passes {

RocmExpansion::RocmExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool RocmExpansion::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();

    bool made_changes = false;

    for (auto* library_node : dataflow.library_nodes()) {
        if (library_node->implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }

        auto& lib_node_code = library_node->code();

        if (lib_node_code == math::tensor::LibraryNodeType_Conv) {
            auto& conv_node = static_cast<math::tensor::ConvNode&>(*library_node);
            sdfg::offloading::RocmConvExpander expander(conv_node);
            expander.expand(builder_, analysis_manager_);
            made_changes = true;
        } else if (lib_node_code == math::tensor::LibraryNodeType_MatMul) {
            auto& matmul_node = static_cast<math::tensor::MatMulNode&>(*library_node);
            sdfg::offloading::RocmBatchedMatMulExpander expander(matmul_node);
            made_changes |= expander.expand(builder_, analysis_manager_);
        } else if (lib_node_code == math::tensor::LibraryNodeType_TensorConcat) {
            auto& concat_node = static_cast<math::tensor::ConcatNode&>(*library_node);
            sdfg::offloading::RocmConcatExpander expander(concat_node);
            made_changes |= expander.expand(builder_, analysis_manager_);
        } else {
            continue;
        }
    }
    return made_changes;
};


} // namespace passes
} // namespace sdfg
