#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include <optional>

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {

std::optional<data_flow::ImplementationType> RocmLibraryNodeRewriter::
    try_library_node_implementation(const data_flow::LibraryNode& lib_node, types::PrimitiveType data_type) {
    if (data_type == types::PrimitiveType::Float || data_type == types::PrimitiveType::Double) {
        if (lib_node.code() == math::blas::LibraryNodeType_GEMM.value()) {
            auto& gemm_node = static_cast<const math::blas::GEMMNode&>(lib_node);
            return try_rocm_gemm_node_implementation(gemm_node, data_type);
        } else if (lib_node.code() == math::blas::LibraryNodeType_DOT.value()) {
            return rocm::ImplementationType_ROCMWithTransfers;
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
}

std::optional<data_flow::ImplementationType> RocmLibraryNodeRewriter::
    try_rocm_gemm_node_implementation(const math::blas::GEMMNode& gemm_node, types::PrimitiveType data_type) {
    // Heuristic: Avoid using ROCm BLAS for very small matrix multiplications
    if (symbolic::eq(gemm_node.m(), symbolic::one()) || symbolic::eq(gemm_node.n(), symbolic::one()) ||
        symbolic::eq(gemm_node.k(), symbolic::one())) {
        return std::nullopt;
    }
    return rocm::ImplementationType_ROCMWithTransfers;
}

std::optional<data_flow::ImplementationType> RocmLibraryNodeRewriter::
    try_memset_implementation(const ::sdfg::stdlib::MemsetNode& memset_node) {
    return rocm::ImplementationType_ROCMWithTransfers;
}

RocmLibraryNodeRewriter::
    RocmLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool RocmLibraryNodeRewriter::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& library_node : dataflow.nodes()) {
        if (auto lib_node = dynamic_cast<math::blas::BLASNode*>(&library_node)) {
            auto implType = try_library_node_implementation(*lib_node, lib_node->scalar_primitive());

            if (implType) {
                lib_node->implementation_type() = implType.value();
            }
        }
        if (auto memset_node = dynamic_cast<::sdfg::stdlib::MemsetNode*>(&library_node)) {
            auto implType = try_memset_implementation(*memset_node);
            if (implType) {
                memset_node->implementation_type() = implType.value();
            }
        }
    }
    return false;
}

} // namespace rocm
} // namespace sdfg
