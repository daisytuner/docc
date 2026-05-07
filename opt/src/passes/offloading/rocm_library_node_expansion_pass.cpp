#include "sdfg/passes/offloading/rocm_library_node_expansion_pass.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/targets/rocm/math/tensor/batchnorm_expander.h"
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

        if (lib_node_code == math::tensor::LibraryNodeType_BatchNorm) {
            auto& batchnorm_node = static_cast<math::tensor::BatchNormNode&>(*library_node);
            sdfg::offloading::RocmBatchNormExpander expander(batchnorm_node);
            expander.expand(builder_, analysis_manager_);
            made_changes = true;
        } else if (lib_node_code == math::tensor::LibraryNodeType_Conv) {
            auto& conv_node = static_cast<math::tensor::ConvNode&>(*library_node);
            sdfg::offloading::RocmConvExpander expander(conv_node);
            expander.expand(builder_, analysis_manager_);
            made_changes = true;
        } else {
            continue;
        }
    }
    return made_changes;
};


} // namespace passes
} // namespace sdfg
