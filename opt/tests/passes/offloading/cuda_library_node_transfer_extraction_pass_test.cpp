#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"
#include "sdfg/targets/cuda/cuda.h"

using namespace sdfg;

TEST(CudaLibraryNodeTransferExtractionPassTest, MemsetExpansion) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);

    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_TRUE(changed);

    // After pass: root should have 3 blocks (alloc, memset, copy+dealloc)
    EXPECT_EQ(sdfg.root().size(), 3);

    // The memset node should now be WithoutTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
}

TEST(CudaLibraryNodeTransferExtractionPassTest, MemsetNoExpansionWhenNone) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);
    // Leave as NONE — pass should not expand

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_FALSE(changed);

    // Should remain 1 block
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(CudaLibraryNodeTransferExtractionPassTest, BatchedGemmExpansion) {
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
            cuda::ImplementationType_CUDAWithTransfers,
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

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_TRUE(changed);

    // After pass: root should have 7 blocks
    // copy_A, copy_B, copy_C, blas_block, copy_C_back, dealloc_A, dealloc_B
    EXPECT_EQ(sdfg.root().size(), 7);

    // The batched gemm node should now be WithoutTransfers
    EXPECT_EQ(batched_gemm_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
}

TEST(CudaLibraryNodeTransferExtractionPassTest, BatchedGemmNoExpansionWhenWrongType) {
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

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_FALSE(changed);

    // Should remain 1 block
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(CudaLibraryNodeTransferExtractionPassTest, SoftmaxExpansion) {
    builder::StructuredSDFGBuilder builder("sdfg_softmax", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);
    std::vector<symbolic::Expression> shape = {symbolic::integer(64), symbolic::integer(128)};
    std::vector<int64_t> axes = {-1};

    builder.add_container("X", ptr_type, true);
    builder.add_container("Y", ptr_type, true);

    auto& block = builder.add_block(sdfg.root());
    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");
    auto& softmax_node =
        static_cast<math::tensor::SoftmaxNode&>(builder.add_library_node<
                                                math::tensor::SoftmaxNode>(block, DebugInfo(), shape, axes, false));
    softmax_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    types::Tensor tensor_type(desc, shape);
    builder.add_computational_memlet(block, y_node, softmax_node, "Y", {}, tensor_type);
    builder.add_computational_memlet(block, x_node, softmax_node, "X", {}, tensor_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_TRUE(changed);

    EXPECT_EQ(sdfg.root().size(), 5);
    EXPECT_EQ(softmax_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
}
