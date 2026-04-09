#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/transformations/offloading/cublas_data_transfer_extraction.h"

using namespace sdfg;

TEST(CUBLASDataTransferExtractionTest, DotCanBeApplied) {
    builder::StructuredSDFGBuilder builder("sdfg_dot", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(1);
    auto stride_b = symbolic::integer(1);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(dot_node);
    EXPECT_TRUE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUBLASDataTransferExtractionTest, DotApply) {
    builder::StructuredSDFGBuilder builder("sdfg_dot", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(1);
    auto stride_b = symbolic::integer(1);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(dot_node);
    ASSERT_TRUE(expansion.can_be_applied(builder, analysis_manager));
    expansion.apply(builder, analysis_manager);

    // After apply: implementation type should be WithoutTransfers
    EXPECT_EQ(dot_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());

    // The root sequence should now have 5 blocks:
    // copy_x_to_device, copy_y_to_device, blas_block, dealloc_x, dealloc_y
    EXPECT_EQ(sdfg.root().size(), 5);

    // The access nodes in the BLAS block should now reference device containers
    EXPECT_NE(a_node.data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
    EXPECT_NE(b_node.data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
}

TEST(CUBLASDataTransferExtractionTest, DotWrongImplType) {
    builder::StructuredSDFGBuilder builder("sdfg_dot", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block, DebugInfo(), cuda::ImplementationType_CUDAWithoutTransfers, math::blas::BLAS_Precision::d, n
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(dot_node);
    EXPECT_FALSE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUBLASDataTransferExtractionTest, GemmCanBeApplied) {
    builder::StructuredSDFGBuilder builder("sdfg_gemm", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_k)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_j)));
    types::Array arr_c_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_j)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("arr_c", arr_c_type);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "arr_a");
    auto& b_node = builder.add_access(block, "arr_b");
    auto& c_in_node = builder.add_access(block, "arr_c");
    auto& c_out_node = builder.add_access(block, "arr_c");
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j),
        symbolic::integer(dim_j)
    ));

    builder.add_computational_memlet(block, a_node, gemm_node, "__A", {symbolic::zero()}, arr_a_type);
    builder.add_computational_memlet(block, b_node, gemm_node, "__B", {symbolic::zero()}, arr_b_type);
    builder.add_computational_memlet(block, c_in_node, gemm_node, "__C", {symbolic::zero()}, arr_c_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);
    builder.add_computational_memlet(block, gemm_node, "__C", c_out_node, {symbolic::zero()}, arr_c_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(gemm_node);
    EXPECT_TRUE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUBLASDataTransferExtractionTest, GemmApply) {
    builder::StructuredSDFGBuilder builder("sdfg_gemm", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_k)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_j)));
    types::Array arr_c_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_j)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("arr_c", arr_c_type);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "arr_a");
    auto& b_node = builder.add_access(block, "arr_b");
    auto& c_in_node = builder.add_access(block, "arr_c");
    auto& c_out_node = builder.add_access(block, "arr_c");
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j),
        symbolic::integer(dim_j)
    ));

    builder.add_computational_memlet(block, a_node, gemm_node, "__A", {symbolic::zero()}, arr_a_type);
    builder.add_computational_memlet(block, b_node, gemm_node, "__B", {symbolic::zero()}, arr_b_type);
    builder.add_computational_memlet(block, c_in_node, gemm_node, "__C", {symbolic::zero()}, arr_c_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);
    builder.add_computational_memlet(block, gemm_node, "__C", c_out_node, {symbolic::zero()}, arr_c_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(gemm_node);
    ASSERT_TRUE(expansion.can_be_applied(builder, analysis_manager));
    expansion.apply(builder, analysis_manager);

    // After apply: implementation type should be WithoutTransfers
    EXPECT_EQ(gemm_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());

    // The root sequence should now have 7 blocks:
    // copy_A_to_device, copy_B_to_device, copy_C_to_device, blas_block,
    // copy_C_from_device, dealloc_A, dealloc_B
    EXPECT_EQ(sdfg.root().size(), 7);
}

TEST(CUBLASDataTransferExtractionTest, GemmWrongImplType) {
    builder::StructuredSDFGBuilder builder("sdfg_gemm", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_k)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_j)));
    types::Array arr_c_type(desc, symbolic::mul(symbolic::integer(dim_i), symbolic::integer(dim_j)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("arr_c", arr_c_type);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "arr_a");
    auto& b_node = builder.add_access(block, "arr_b");
    auto& c_in_node = builder.add_access(block, "arr_c");
    auto& c_out_node = builder.add_access(block, "arr_c");
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithoutTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j),
        symbolic::integer(dim_j)
    ));

    builder.add_computational_memlet(block, a_node, gemm_node, "__A", {symbolic::zero()}, arr_a_type);
    builder.add_computational_memlet(block, b_node, gemm_node, "__B", {symbolic::zero()}, arr_b_type);
    builder.add_computational_memlet(block, c_in_node, gemm_node, "__C", {symbolic::zero()}, arr_c_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);
    builder.add_computational_memlet(block, gemm_node, "__C", c_out_node, {symbolic::zero()}, arr_c_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUBLASDataTransferExtraction expansion(gemm_node);
    EXPECT_FALSE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUBLASDataTransferExtractionTest, DotSerialization) {
    builder::StructuredSDFGBuilder builder("sdfg_dot", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block, DebugInfo(), cuda::ImplementationType_CUDAWithTransfers, math::blas::BLAS_Precision::d, n
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc);
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc);

    cuda::CUBLASDataTransferExtraction expansion(dot_node);

    nlohmann::json j;
    expansion.to_json(j);

    EXPECT_EQ(j["transformation_type"], "CUBLASDataTransferExtraction");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], dot_node.element_id());

    auto deserialized = cuda::CUBLASDataTransferExtraction::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "CUBLASDataTransferExtraction");
}

TEST(CUBLASDataTransferExtractionTest, GemmSerialization) {
    builder::StructuredSDFGBuilder builder("sdfg_gemm", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(300));
    types::Array arr_b_type(desc, symbolic::integer(600));
    types::Array arr_c_type(desc, symbolic::integer(200));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("arr_c", arr_c_type);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "arr_a");
    auto& b_node = builder.add_access(block, "arr_b");
    auto& c_in_node = builder.add_access(block, "arr_c");
    auto& c_out_node = builder.add_access(block, "arr_c");
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(10),
        symbolic::integer(20),
        symbolic::integer(30),
        symbolic::integer(30),
        symbolic::integer(20),
        symbolic::integer(20)
    ));

    builder.add_computational_memlet(block, a_node, gemm_node, "__A", {symbolic::zero()}, arr_a_type);
    builder.add_computational_memlet(block, b_node, gemm_node, "__B", {symbolic::zero()}, arr_b_type);
    builder.add_computational_memlet(block, c_in_node, gemm_node, "__C", {symbolic::zero()}, arr_c_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);
    builder.add_computational_memlet(block, gemm_node, "__C", c_out_node, {symbolic::zero()}, arr_c_type);

    cuda::CUBLASDataTransferExtraction expansion(gemm_node);

    nlohmann::json j;
    expansion.to_json(j);

    EXPECT_EQ(j["transformation_type"], "CUBLASDataTransferExtraction");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], gemm_node.element_id());

    auto deserialized = cuda::CUBLASDataTransferExtraction::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "CUBLASDataTransferExtraction");
}
