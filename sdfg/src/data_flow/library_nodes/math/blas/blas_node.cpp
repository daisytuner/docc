#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"

namespace sdfg::math::blas {

BLASNode::BLASNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const data_flow::ImplementationType& implementation_type,
    const BLAS_Precision& precision
)
    : MathNode(element_id, debug_info, vertex, parent, code, outputs, inputs, implementation_type, true),
      precision_(precision) {}

types::PrimitiveType BLASNode::scalar_primitive() const {
    switch (this->precision_) {
        case BLAS_Precision::s:
            return types::PrimitiveType::Float;
        case BLAS_Precision::d:
            return types::PrimitiveType::Double;
        case BLAS_Precision::h:
            return types::PrimitiveType::Half;
        default:
            return types::PrimitiveType::Void;
    }
}

BLAS_Precision BLASNode::precision() const { return this->precision_; };


} // namespace sdfg::math::blas
