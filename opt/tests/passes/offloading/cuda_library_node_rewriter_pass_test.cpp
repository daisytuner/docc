#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include "sdfg/targets/cuda/cuda.h"

using namespace sdfg;

TEST(CudaLibraryNodeRewriterPassTest, MemsetRewrite) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);

    // Before rewriter: implementation type should be NONE
    EXPECT_EQ(memset_node.implementation_type().value(), data_flow::ImplementationType_NONE.value());

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    // After rewriter: implementation type should be CUDAWithTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), cuda::ImplementationType_CUDAWithTransfers.value());
}

TEST(CudaLibraryNodeRewriterPassTest, MemsetRewriteDoublePrecision) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(2048);
    auto value = symbolic::integer(255);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    // Memset rewrite should work regardless of scalar type
    EXPECT_EQ(memset_node.implementation_type().value(), cuda::ImplementationType_CUDAWithTransfers.value());
}

TEST(CudaLibraryNodeRewriterPassTest, BatchedGemmRewrite) {
    builder::StructuredSDFGBuilder builder("sdfg_batched_gemm", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_batch = 4;
    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(dim_batch * dim_i * dim_k));
    types::Array arr_b_type(desc, symbolic::integer(dim_batch * dim_k * dim_j));
    types::Array arr_c_type(desc, symbolic::integer(dim_batch * dim_i * dim_j));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("arr_c", arr_c_type);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "arr_a");
    auto& b_node = builder.add_access(block, "arr_b");
    auto& c_node = builder.add_access(block, "arr_c");
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& batched_gemm_node =
        static_cast<math::blas::BatchedGEMMNode&>(builder.add_library_node<math::blas::BatchedGEMMNode>(
            block,
            DebugInfo(),
            data_flow::ImplementationType_NONE,
            math::blas::BLAS_Precision::s,
            math::blas::BLAS_Layout::RowMajor,
            math::blas::BLAS_Transpose::No,
            math::blas::BLAS_Transpose::No,
            symbolic::integer(dim_batch),
            symbolic::integer(dim_i),
            symbolic::integer(dim_j),
            symbolic::integer(dim_k),
            symbolic::integer(dim_k),
            symbolic::integer(dim_j),
            symbolic::integer(dim_j),
            symbolic::integer(dim_i * dim_k),
            symbolic::integer(dim_k * dim_j),
            symbolic::integer(dim_i * dim_j)
        ));

    builder.add_computational_memlet(block, a_node, batched_gemm_node, "__A", {symbolic::zero()}, arr_a_type);
    builder.add_computational_memlet(block, b_node, batched_gemm_node, "__B", {symbolic::zero()}, arr_b_type);
    builder.add_computational_memlet(block, c_node, batched_gemm_node, "__C", {symbolic::zero()}, arr_c_type);
    builder.add_computational_memlet(block, alpha_node, batched_gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, batched_gemm_node, "__beta", {}, desc);

    // Before rewriter: implementation type should be NONE
    EXPECT_EQ(batched_gemm_node.implementation_type().value(), data_flow::ImplementationType_NONE.value());

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    // After rewriter: implementation type should be CUDAWithTransfers
    EXPECT_EQ(batched_gemm_node.implementation_type().value(), cuda::ImplementationType_CUDAWithTransfers.value());
}
