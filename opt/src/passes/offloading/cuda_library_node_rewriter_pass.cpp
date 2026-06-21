#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include <optional>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace cuda {

std::optional<data_flow::ImplementationType> CudaLibraryNodeRewriter::
    try_library_node_implementation(const data_flow::LibraryNode& lib_node, types::PrimitiveType data_type) {
    if (data_type == types::PrimitiveType::Float || data_type == types::PrimitiveType::Double) {
        if (lib_node.code() == math::blas::LibraryNodeType_GEMM.value()) {
            auto& gemm_node = static_cast<const math::blas::GEMMNode&>(lib_node);
            return try_cublas_gemm_node_implementation(gemm_node, data_type);
        } else if (lib_node.code() == math::blas::LibraryNodeType_BatchedGEMM.value()) {
            return cuda::ImplementationType_CUDAWithTransfers;
        } else if (lib_node.code() == math::blas::LibraryNodeType_DOT.value()) {
            return cuda::ImplementationType_CUDAWithTransfers;
        } else if (lib_node.code() == math::blas::LibraryNodeType_BatchedGEMM.value()) {
            auto& batched_gemm_node = static_cast<const math::blas::BatchedGEMMNode&>(lib_node);
            return try_cublas_batched_gemm_node_implementation(batched_gemm_node, data_type);
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
}

std::optional<data_flow::ImplementationType> CudaLibraryNodeRewriter::
    try_cublas_gemm_node_implementation(const math::blas::GEMMNode& gemm_node, types::PrimitiveType data_type) {
    return cuda::ImplementationType_CUDAWithTransfers;
}

std::optional<data_flow::ImplementationType> CudaLibraryNodeRewriter::
    try_memset_implementation(const ::sdfg::stdlib::MemsetNode& memset_node) {
    return cuda::ImplementationType_CUDAWithTransfers;
}

std::optional<data_flow::ImplementationType> CudaLibraryNodeRewriter::
    try_memcpy_implementation(const ::sdfg::stdlib::MemcpyNode& memcpy_node) {
    return cuda::ImplementationType_CUDAWithTransfers;
}

std::optional<data_flow::ImplementationType> CudaLibraryNodeRewriter::try_cublas_batched_gemm_node_implementation(
    const math::blas::BatchedGEMMNode& batched_gemm_node, types::PrimitiveType data_type
) {
    // Heuristic: Avoid using CUBLAS for very small matrix multiplications
    auto m = batched_gemm_node.m();
    auto n = batched_gemm_node.n();
    auto k = batched_gemm_node.k();
    auto size = symbolic::mul(symbolic::mul(m, n), k);
    if (symbolic::eq(size, symbolic::one())) {
        return std::nullopt;
    }
    return cuda::ImplementationType_CUDAWithTransfers;
}

CudaLibraryNodeRewriter::
    CudaLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool CudaLibraryNodeRewriter::accept(structured_control_flow::Block& node) {
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
        if (auto memcpy_node = dynamic_cast<::sdfg::stdlib::MemcpyNode*>(&library_node)) {
            auto implType = try_memcpy_implementation(*memcpy_node);
            if (implType) {
                memcpy_node->implementation_type() = implType.value();
            }
        }
    }
    return false;
}

} // namespace cuda
} // namespace sdfg
