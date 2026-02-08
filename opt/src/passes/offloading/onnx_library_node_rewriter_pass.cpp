#include "sdfg/passes/offloading/onnx_library_node_rewriter_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace passes {

ONNXLibraryNodeRewriter::
    ONNXLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool ONNXLibraryNodeRewriter::visit() {
    DEBUG_PRINTLN("Running ONNXLibraryNodeRewriterPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool ONNXLibraryNodeRewriter::accept(structured_control_flow::Block& block) {
    bool applied = false;

    for (auto* lib_node : block.dataflow().library_nodes()) {
        if (auto* tensor_node = dynamic_cast<math::tensor::TensorNode*>(lib_node)) {
            tensor_node->implementation_type() = onnx::ImplementationType_ONNX;
            applied = true;
        } else if (auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(lib_node)) {
            gemm_node->implementation_type() = onnx::ImplementationType_ONNX;
            applied = true;
        } else if (auto* dot_node = dynamic_cast<math::blas::DotNode*>(lib_node)) {
            dot_node->implementation_type() = onnx::ImplementationType_ONNX;
            applied = true;
        }
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
