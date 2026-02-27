#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"

using namespace sdfg;

TEST(MatMulTest, MatMul_2D_SimpleMatrix) {
    // Test simple 2D matrix multiplication: A[M, K] @ B[K, N] = Y[M, N]
    builder::StructuredSDFGBuilder builder("sdfg_matmul_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [4, 8] (M=4, K=8)
    // Shape B: [8, 6] (K=8, N=6)
    // Output Y: [4, 6] (M=4, N=6)
    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check basic properties
    EXPECT_EQ(matmul_node.inputs().size(), 2);
    EXPECT_EQ(matmul_node.inputs()[0], "A");
    EXPECT_EQ(matmul_node.inputs()[1], "B");
    EXPECT_EQ(matmul_node.outputs().size(), 1);
    EXPECT_EQ(matmul_node.outputs()[0], "Y");

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    EXPECT_EQ(block.dataflow().nodes().size(), 4); // a, b, y, matmul

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // After expansion, the root should contain a new sequence
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // The sequence should contain:
    // 1. Reference block (creating references for batch offsets)
    // 2. GEMM block
    // Since no batch dimensions, just 2 blocks
    EXPECT_GE(new_sequence.size(), 2);

    // Find the GEMM block - it should contain the GEMMNode
    bool found_gemm = false;
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(i).first);
        if (blk) {
            for (auto& node : blk->dataflow().library_nodes()) {
                if (auto* gemm = dynamic_cast<math::blas::GEMMNode*>(node)) {
                    found_gemm = true;

                    // Check GEMM parameters
                    EXPECT_TRUE(symbolic::eq(gemm->m(), symbolic::integer(4)));
                    EXPECT_TRUE(symbolic::eq(gemm->n(), symbolic::integer(6)));
                    EXPECT_TRUE(symbolic::eq(gemm->k(), symbolic::integer(8)));
                    EXPECT_EQ(gemm->layout(), math::blas::BLAS_Layout::RowMajor);
                    EXPECT_EQ(gemm->trans_a(), math::blas::BLAS_Transpose::No);
                    EXPECT_EQ(gemm->trans_b(), math::blas::BLAS_Transpose::No);
                    EXPECT_EQ(gemm->precision(), math::blas::BLAS_Precision::s);

                    // Check GEMM has correct connections
                    auto& dataflow = gemm->get_parent();
                    EXPECT_EQ(dataflow.in_degree(*gemm), 5); // A, B, C, alpha, beta
                    EXPECT_EQ(dataflow.out_degree(*gemm), 1); // C output
                }
            }
        }
    }
    EXPECT_TRUE(found_gemm);
}

TEST(MatMulTest, MatMul_3D_Batched) {
    // Test batched matrix multiplication: A[B, M, K] @ B[B, K, N] = Y[B, M, N]
    builder::StructuredSDFGBuilder builder("sdfg_matmul_3d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [2, 4, 8] (B=2, M=4, K=8)
    // Shape B: [2, 8, 6] (B=2, K=8, N=6)
    // Output Y: [2, 4, 6] (B=2, M=4, N=6)
    symbolic::MultiExpression shape_a = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(2), symbolic::integer(8), symbolic::integer(6)};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(2), symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // After expansion, should have a map for the batch dimension
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Look for a Map (batch loop)
    bool found_batch_map = false;
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        if (dynamic_cast<structured_control_flow::Map*>(&new_sequence.at(i).first)) {
            found_batch_map = true;
            break;
        }
    }
    EXPECT_TRUE(found_batch_map);
}

TEST(MatMulTest, MatMul_WithSymbolicDimensions) {
    // Test matmul with symbolic dimensions
    builder::StructuredSDFGBuilder builder("sdfg_matmul_symbolic", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);
    builder.add_container("M", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("K", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");

    symbolic::MultiExpression shape_a = {M, K};
    symbolic::MultiExpression shape_b = {K, N};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {M, N};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check dimensions are symbolic
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), M));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), N));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), K));

    // Check symbols() returns all symbolic dimensions
    auto syms = matmul_node.symbols();
    EXPECT_TRUE(syms.find(M) != syms.end());
    EXPECT_TRUE(syms.find(N) != syms.end());
    EXPECT_TRUE(syms.find(K) != syms.end());

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // Verify GEMM node has symbolic dimensions
    bool found_gemm = false;
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(i).first);
        if (blk) {
            for (auto& node : blk->dataflow().library_nodes()) {
                if (auto* gemm = dynamic_cast<math::blas::GEMMNode*>(node)) {
                    found_gemm = true;
                    EXPECT_TRUE(symbolic::eq(gemm->m(), M));
                    EXPECT_TRUE(symbolic::eq(gemm->n(), N));
                    EXPECT_TRUE(symbolic::eq(gemm->k(), K));
                }
            }
        }
    }
    EXPECT_TRUE(found_gemm);
}

TEST(MatMulTest, MatMul_WithStridesAndOffset) {
    // Test matmul with custom strides and offsets
    builder::StructuredSDFGBuilder builder("sdfg_matmul_strides", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [4, 8], custom strides and offset
    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};

    // Custom strides (column-major for A)
    symbolic::MultiExpression strides_a = {symbolic::integer(1), symbolic::integer(4)};
    symbolic::MultiExpression strides_b = {}; // Default row-major

    // Offset
    auto offset_a = symbolic::integer(16);
    auto offset_b = symbolic::integer(0);

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), shape_a, shape_b, strides_a, strides_b, offset_a, offset_b
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    // Check strides and offset
    EXPECT_TRUE(symbolic::eq(matmul_node.strides_a()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(matmul_node.strides_a()[1], symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.offset_a(), symbolic::integer(16)));
    EXPECT_TRUE(symbolic::eq(matmul_node.offset_b(), symbolic::integer(0)));

    sdfg.validate();

    // Test expansion - should create a copy since strides are non-contiguous
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // Verify the expanded structure contains malloc before GEMM and free after
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Track positions of malloc, gemm, and free
    int malloc_position = -1;
    int gemm_position = -1;
    int free_position = -1;

    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(i).first);
        if (blk) {
            for (auto& node : blk->dataflow().library_nodes()) {
                if (dynamic_cast<stdlib::MallocNode*>(node)) {
                    malloc_position = static_cast<int>(i);
                }
                if (dynamic_cast<math::blas::GEMMNode*>(node)) {
                    gemm_position = static_cast<int>(i);
                }
                if (dynamic_cast<stdlib::FreeNode*>(node)) {
                    free_position = static_cast<int>(i);
                }
            }
        }
    }

    // Verify malloc was found (since A has non-default strides)
    EXPECT_NE(malloc_position, -1) << "Expected MallocNode for view copy";

    // Verify GEMM was found
    EXPECT_NE(gemm_position, -1) << "Expected GEMMNode in expansion";

    // Verify free was found
    EXPECT_NE(free_position, -1) << "Expected FreeNode for view copy cleanup";

    // Verify order: malloc < gemm < free
    EXPECT_LT(malloc_position, gemm_position) << "MallocNode should come before GEMMNode";
    EXPECT_LT(gemm_position, free_position) << "FreeNode should come after GEMMNode";
}

TEST(MatMulTest, MatMul_IntegerType_ReturnsFailure) {
    // Test that integer types are not supported (GEMM requires float)
    builder::StructuredSDFGBuilder builder("sdfg_matmul_int", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node =
        static_cast<math::tensor::MatMulNode&>(builder.add_library_node<
                                               math::tensor::MatMulNode>(block, DebugInfo(), shape_a, shape_b));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Test expansion - should fail for integer types
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_FALSE(matmul_node.expand(builder, analysis_manager));
}
TEST(MatMulTest, MatMul_BothViewsCopied) {
    // Test matmul where both A and B have non-default strides (both are views)
    builder::StructuredSDFGBuilder builder("sdfg_matmul_both_views", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [4, 8], Shape B: [8, 6]
    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};

    // Non-default strides for both (column-major layout)
    symbolic::MultiExpression strides_a = {symbolic::integer(1), symbolic::integer(4)};
    symbolic::MultiExpression strides_b = {symbolic::integer(1), symbolic::integer(8)};

    // Offsets
    auto offset_a = symbolic::integer(0);
    auto offset_b = symbolic::integer(0);

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), shape_a, shape_b, strides_a, strides_b, offset_a, offset_b
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // Verify the expanded structure contains 2 mallocs before GEMM and 2 frees after
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Count and track positions of malloc, gemm, and free nodes
    std::vector<int> malloc_positions;
    std::vector<int> free_positions;
    int gemm_position = -1;

    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(i).first);
        if (blk) {
            for (auto& node : blk->dataflow().library_nodes()) {
                if (dynamic_cast<stdlib::MallocNode*>(node)) {
                    malloc_positions.push_back(static_cast<int>(i));
                }
                if (dynamic_cast<math::blas::GEMMNode*>(node)) {
                    gemm_position = static_cast<int>(i);
                }
                if (dynamic_cast<stdlib::FreeNode*>(node)) {
                    free_positions.push_back(static_cast<int>(i));
                }
            }
        }
    }

    // Verify two mallocs were found (one for A view, one for B view)
    EXPECT_EQ(malloc_positions.size(), 2) << "Expected 2 MallocNodes for both view copies";

    // Verify GEMM was found
    EXPECT_NE(gemm_position, -1) << "Expected GEMMNode in expansion";

    // Verify two frees were found
    EXPECT_EQ(free_positions.size(), 2) << "Expected 2 FreeNodes for both view copy cleanups";

    // Verify order: all mallocs < gemm < all frees
    for (int malloc_pos : malloc_positions) {
        EXPECT_LT(malloc_pos, gemm_position) << "All MallocNodes should come before GEMMNode";
    }
    for (int free_pos : free_positions) {
        EXPECT_GT(free_pos, gemm_position) << "All FreeNodes should come after GEMMNode";
    }
}

TEST(MatMulTest, MatMul_NoCopyForDefaultStrides) {
    // Test that no malloc/free are created when strides are default (row-major)
    builder::StructuredSDFGBuilder builder("sdfg_matmul_no_copy", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("y", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    // Shape A: [4, 8], Shape B: [8, 6]
    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};

    // Default row-major strides (empty = auto-computed)
    symbolic::MultiExpression strides_a = {};
    symbolic::MultiExpression strides_b = {};

    // No offset
    auto offset_a = symbolic::integer(0);
    auto offset_b = symbolic::integer(0);

    types::Tensor input_tensor_a(desc.primitive_type(), shape_a);
    types::Tensor input_tensor_b(desc.primitive_type(), shape_b);
    symbolic::MultiExpression output_shape = {symbolic::integer(4), symbolic::integer(6)};
    types::Tensor output_tensor(desc.primitive_type(), output_shape);

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), shape_a, shape_b, strides_a, strides_b, offset_a, offset_b
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, matmul_node, "Y", y_node, {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(matmul_node.expand(builder, analysis_manager));

    // Verify no malloc or free nodes exist (no copies needed)
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    int malloc_count = 0;
    int free_count = 0;
    bool found_gemm = false;

    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(i).first);
        if (blk) {
            for (auto& node : blk->dataflow().library_nodes()) {
                if (dynamic_cast<stdlib::MallocNode*>(node)) {
                    malloc_count++;
                }
                if (dynamic_cast<math::blas::GEMMNode*>(node)) {
                    found_gemm = true;
                }
                if (dynamic_cast<stdlib::FreeNode*>(node)) {
                    free_count++;
                }
            }
        }
    }

    // No malloc or free should be needed for default strides
    EXPECT_EQ(malloc_count, 0) << "No MallocNode expected for default strides";
    EXPECT_EQ(free_count, 0) << "No FreeNode expected for default strides";
    EXPECT_TRUE(found_gemm) << "GEMMNode should still be present";
}
