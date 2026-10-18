#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg_debug_dump.h"

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

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b)
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, y_node, matmul_node, "Y", {}, output_tensor, block.debug_info());

    dump_sdfg(sdfg, "0.before");

    // Check basic properties
    EXPECT_EQ(matmul_node.inputs().size(), 3);
    EXPECT_EQ(matmul_node.inputs()[math::tensor::MatMulNode::Y_INPUT_IDX], "Y");
    EXPECT_EQ(matmul_node.inputs()[math::tensor::MatMulNode::A_INPUT_IDX], "A");
    EXPECT_EQ(matmul_node.inputs()[math::tensor::MatMulNode::B_INPUT_IDX], "B");
    EXPECT_EQ(matmul_node.outputs().size(), 0);

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    EXPECT_EQ(block.dataflow().nodes().size(), 4); // a, b, y, matmul

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    auto outcome =
        passes::expansion::expand_single_node(builder, block, matmul_node, math::tensor::MatMulBlasExpander());
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    dump_sdfg(sdfg, "1.expanded");

    // After expansion, the root should contain a new sequence
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));

    // The sequence should contain:
    // 1. Reference block (creating references for batch offsets)
    // 2. GEMM block
    // Since no batch dimensions, just 2 blocks
    EXPECT_GE(new_sequence.size(), 2);

    // Find the GEMM block - it should contain the GEMMNode
    bool found_gemm = false;
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&new_sequence.at(i));
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
                    EXPECT_EQ(dataflow.out_degree(*gemm), 0); // no output edge, C is written via the ptr
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

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b)
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, y_node, matmul_node, "Y", {}, output_tensor, block.debug_info());

    dump_sdfg(sdfg, "0.init");

    // Check dimensions
    EXPECT_TRUE(symbolic::eq(matmul_node.m(), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(matmul_node.n(), symbolic::integer(6)));
    EXPECT_TRUE(symbolic::eq(matmul_node.k(), symbolic::integer(8)));

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    auto outcome =
        passes::expansion::expand_single_node(builder, block, matmul_node, math::tensor::MatMulBlasExpander());

    dump_sdfg(sdfg, "1.expanded");
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    // After expansion, should have a map for the batch dimension
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));

    // Look for a Map (batch loop)
    bool found_batch_map = false;
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        if (dyn_cast<structured_control_flow::Map*>(&new_sequence.at(i))) {
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

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b)
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, y_node, matmul_node, "Y", {}, output_tensor, block.debug_info());

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
    auto outcome =
        passes::expansion::expand_single_node(builder, block, matmul_node, math::tensor::MatMulBlasExpander());
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    // Verify GEMM node has symbolic dimensions
    bool found_gemm = false;
    auto& new_sequence = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));
    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&new_sequence.at(i));
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

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b)
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, y_node, matmul_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Test expansion - should fail for integer types
    analysis::AnalysisManager analysis_manager(sdfg);
    auto outcome =
        passes::expansion::expand_single_node(builder, block, matmul_node, math::tensor::MatMulBlasExpander());
    EXPECT_FALSE(outcome.expanded);
    EXPECT_FALSE(outcome.block_removed);
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
        block,
        DebugInfo(),
        math::tensor::TensorLayout(shape_a, strides_a, offset_a),
        math::tensor::TensorLayout(shape_b, strides_b, offset_b)
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, input_tensor_a, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, input_tensor_b, block.debug_info());
    builder.add_computational_memlet(block, y_node, matmul_node, "Y", {}, output_tensor, block.debug_info());

    sdfg.validate();

    // Test expansion
    analysis::AnalysisManager analysis_manager(sdfg);
    auto outcome =
        passes::expansion::expand_single_node(builder, block, matmul_node, math::tensor::MatMulBlasExpander());
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    // Verify no malloc or free nodes exist (no copies needed)
    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));

    int malloc_count = 0;
    int free_count = 0;
    bool found_gemm = false;

    for (size_t i = 0; i < new_sequence.size(); ++i) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&new_sequence.at(i));
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

// The MatMul node reports each operand's consumed affine layout through its
// pointer access metadata, so MemoryLayoutAnalysis can localize a whole operand.
TEST(MatMulTest, MatMul_ConsumedLayout) {
    builder::StructuredSDFGBuilder builder("matmul_consumed_layout", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& block = builder.add_block(sdfg.root());

    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)}; // M=4, K=8
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)}; // K=8, N=6
    auto& mm = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b)
    ));

    auto a = mm.pointer_access_type(math::tensor::MatMulNode::A_INPUT_IDX);
    ASSERT_NE(a, nullptr);
    EXPECT_TRUE(a->may_contain_reads());
    EXPECT_FALSE(a->may_contain_writes());
    auto a_read = a->access_read_pattern();
    ASSERT_NE(a_read, nullptr);
    ASSERT_NE(a_read->layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(a_read->layout()->get_dim(0), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(a_read->layout()->get_dim(1), symbolic::integer(8)));

    auto b = mm.pointer_access_type(math::tensor::MatMulNode::B_INPUT_IDX);
    auto b_read = b->access_read_pattern();
    ASSERT_NE(b_read, nullptr);
    ASSERT_NE(b_read->layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(b_read->layout()->get_dim(0), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(b_read->layout()->get_dim(1), symbolic::integer(6)));

    auto y = mm.pointer_access_type(math::tensor::MatMulNode::Y_INPUT_IDX);
    ASSERT_NE(y, nullptr);
    EXPECT_TRUE(y->may_contain_writes());
    auto y_write = y->access_write_pattern();
    ASSERT_NE(y_write, nullptr);
    ASSERT_NE(y_write->layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(y_write->layout()->get_dim(0), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(y_write->layout()->get_dim(1), symbolic::integer(6)));
}

namespace {

/// Y = A @ B with A[4, 8], B[8, 6] of element type @p input and Y[4, 6] of @p output.
math::tensor::MatMulNode& add_typed_matmul(
    builder::StructuredSDFGBuilder& builder,
    types::PrimitiveType input,
    types::PrimitiveType output,
    const math::tensor::ContractionTypes& node_types
) {
    auto& sdfg = builder.subject();
    builder.add_container("a", types::Pointer(types::Scalar(input)), true);
    builder.add_container("b", types::Pointer(types::Scalar(input)), true);
    builder.add_container("y", types::Pointer(types::Scalar(output)), true);
    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& y_node = builder.add_access(block, "y");

    symbolic::MultiExpression shape_a = {symbolic::integer(4), symbolic::integer(8)};
    symbolic::MultiExpression shape_b = {symbolic::integer(8), symbolic::integer(6)};
    symbolic::MultiExpression shape_y = {symbolic::integer(4), symbolic::integer(6)};
    auto& matmul = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape_a), math::tensor::TensorLayout(shape_b), node_types
    ));
    builder.add_computational_memlet(block, a_node, matmul, "A", {}, types::Tensor(input, shape_a));
    builder.add_computational_memlet(block, b_node, matmul, "B", {}, types::Tensor(input, shape_b));
    builder.add_computational_memlet(block, y_node, matmul, "Y", {}, types::Tensor(output, shape_y));
    return matmul;
}

structured_control_flow::Block& block_of(math::tensor::MatMulNode& node) {
    return static_cast<structured_control_flow::Block&>(*node.get_parent().get_parent());
}

bool has_tasklet(structured_control_flow::ControlFlowNode& node, data_flow::TaskletCode code) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto* tasklet : block->dataflow().tasklets()) {
            if (tasklet->code() == code) {
                return true;
            }
        }
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            if (has_tasklet(sequence->at(i), code)) {
                return true;
            }
        }
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return has_tasklet(loop->root(), code);
    }
    return false;
}

/// The first serialized library node with code @p code in @p j.
nlohmann::json* find_library_node(nlohmann::json& j, const std::string& code) {
    if (j.is_object()) {
        if (j.contains("code") && j["code"].is_string() && j["code"] == code) {
            return &j;
        }
        for (auto& [_, value] : j.items()) {
            if (auto* found = find_library_node(value, code)) {
                return found;
            }
        }
    } else if (j.is_array()) {
        for (auto& value : j) {
            if (auto* found = find_library_node(value, code)) {
                return found;
            }
        }
    }
    return nullptr;
}

math::tensor::MatMulNode* find_matmul(structured_control_flow::Sequence& root) {
    for (size_t i = 0; i < root.size(); i++) {
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&root.at(i))) {
            for (auto* node : block->dataflow().library_nodes()) {
                if (auto* matmul = dynamic_cast<math::tensor::MatMulNode*>(node)) {
                    return matmul;
                }
            }
        }
    }
    return nullptr;
}

} // namespace

TEST(MatMulTest, MatMul_Int8ToInt32_ExplicitTypes) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_int8", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& matmul = add_typed_matmul(
        builder,
        types::PrimitiveType::Int8,
        types::PrimitiveType::Int32,
        {types::PrimitiveType::Int8, types::PrimitiveType::Void, types::PrimitiveType::Int32}
    );
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(matmul.input_type(), types::PrimitiveType::Int8);
    EXPECT_EQ(matmul.accumulator_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(matmul.output_type(), types::PrimitiveType::Int32);
    EXPECT_FALSE(matmul.uniform_quantization(matmul.get_parent()).has_value());

    // BLAS has no mixed-type GEMM; the naive expansion accumulates in Int32.
    auto& block = block_of(matmul);
    auto blas = passes::expansion::expand_single_node(builder, block, matmul, math::tensor::MatMulBlasExpander());
    EXPECT_FALSE(blas.expanded);

    auto naive = passes::expansion::expand_single_node(builder, block, matmul, math::tensor::MatMulNaiveExpander());
    EXPECT_TRUE(naive.expanded);
    EXPECT_NO_THROW(sdfg.validate());
    EXPECT_TRUE(has_tasklet(sdfg.root(), data_flow::TaskletCode::int_mul));
    EXPECT_TRUE(has_tasklet(sdfg.root(), data_flow::TaskletCode::int_add));
}

TEST(MatMulTest, MatMul_MixedTypes_Inferred) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_inferred", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& matmul = add_typed_matmul(
        builder, types::PrimitiveType::Int8, types::PrimitiveType::Int32, math::tensor::ContractionTypes{}
    );
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(matmul.input_type(), types::PrimitiveType::Int8);
    EXPECT_EQ(matmul.accumulator_type(), types::PrimitiveType::Int32);
    EXPECT_EQ(matmul.output_type(), types::PrimitiveType::Int32);
}

TEST(MatMulTest, MatMul_Uniform_Inferred) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_uniform", FunctionType_CPU);
    auto& matmul = add_typed_matmul(
        builder, types::PrimitiveType::Float, types::PrimitiveType::Float, math::tensor::ContractionTypes{}
    );
    EXPECT_EQ(matmul.uniform_quantization(matmul.get_parent()), types::PrimitiveType::Float);
}

TEST(MatMulTest, MatMul_InputTypeMismatch_Throws) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_mismatch", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_typed_matmul(
        builder,
        types::PrimitiveType::Float,
        types::PrimitiveType::Int32,
        {types::PrimitiveType::Int8, types::PrimitiveType::Void, types::PrimitiveType::Int32}
    );
    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(MatMulTest, MatMul_WiderAccumulator_NotExpandedInPlace) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_wide_acc", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& matmul = add_typed_matmul(
        builder,
        types::PrimitiveType::Half,
        types::PrimitiveType::Half,
        {types::PrimitiveType::Half, types::PrimitiveType::Float, types::PrimitiveType::Half}
    );
    EXPECT_NO_THROW(sdfg.validate());
    EXPECT_EQ(matmul.accumulator_type(), types::PrimitiveType::Float);
    EXPECT_FALSE(matmul.uniform_quantization(matmul.get_parent()).has_value());

    auto naive =
        passes::expansion::expand_single_node(builder, block_of(matmul), matmul, math::tensor::MatMulNaiveExpander());
    EXPECT_FALSE(naive.expanded);
}

TEST(MatMulTest, MatMul_Types_SerializationRoundTrip) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_serialize", FunctionType_CPU);
    math::tensor::ContractionTypes node_types{
        types::PrimitiveType::Int8, types::PrimitiveType::Int32, types::PrimitiveType::Int32
    };
    add_typed_matmul(builder, types::PrimitiveType::Int8, types::PrimitiveType::Int32, node_types);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto* node_json = find_library_node(j, math::tensor::LibraryNodeType_MatMul.value());
    ASSERT_NE(node_json, nullptr);
    // Older readers see no single type.
    EXPECT_EQ((*node_json)["result_quant"].get<types::PrimitiveType>(), types::PrimitiveType::Void);

    auto restored = serializer.deserialize(j);
    auto* matmul = find_matmul(restored->root());
    ASSERT_NE(matmul, nullptr);
    EXPECT_EQ(matmul->types(), node_types);
}

TEST(MatMulTest, MatMul_Types_LegacyDeserialization) {
    builder::StructuredSDFGBuilder builder("sdfg_matmul_legacy", FunctionType_CPU);
    add_typed_matmul(
        builder,
        types::PrimitiveType::Float,
        types::PrimitiveType::Float,
        math::tensor::ContractionTypes::uniform(types::PrimitiveType::Float)
    );

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto* node_json = find_library_node(j, math::tensor::LibraryNodeType_MatMul.value());
    ASSERT_NE(node_json, nullptr);
    EXPECT_EQ((*node_json)["result_quant"].get<types::PrimitiveType>(), types::PrimitiveType::Float);
    node_json->erase("input_type");
    node_json->erase("accumulator_type");
    node_json->erase("output_type");

    auto restored = serializer.deserialize(j);
    auto* matmul = find_matmul(restored->root());
    ASSERT_NE(matmul, nullptr);
    EXPECT_EQ(matmul->types(), math::tensor::ContractionTypes::uniform(types::PrimitiveType::Float));
}

TEST(MatMulTest, MatMul_Naive_PointerMemletsUseLayouts) {
    // Pointer memlets, as einsum detection creates them: the layouts address the operands, here B transposed.
    builder::StructuredSDFGBuilder builder("sdfg_matmul_naive_pointer", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Pointer pointer{types::Scalar(types::PrimitiveType::Int32)};
    builder.add_container("a", pointer, true);
    builder.add_container("b", pointer, true);
    builder.add_container("y", pointer, true);
    auto& block = builder.add_block(sdfg.root());
    auto m = symbolic::integer(4), n = symbolic::integer(6), k = symbolic::integer(8);
    math::tensor::TensorLayout layout_b_transposed({k, n}, {symbolic::one(), k}, symbolic::zero());
    auto& matmul = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout({m, k}), layout_b_transposed
    ));
    builder.add_computational_memlet(block, builder.add_access(block, "a"), matmul, "A", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), matmul, "B", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "y"), matmul, "Y", {}, pointer);
    EXPECT_NO_THROW(sdfg.validate());

    auto outcome = passes::expansion::expand_single_node(builder, block, matmul, math::tensor::MatMulNaiveExpander());
    ASSERT_TRUE(outcome.expanded);
    EXPECT_NO_THROW(sdfg.validate());

    // Every access to b is linear, k + 8 j for the transposed layout.
    std::function<void(structured_control_flow::ControlFlowNode&)> check = [&](auto& node) {
        if (auto* inner = dynamic_cast<structured_control_flow::Block*>(&node)) {
            for (auto& memlet : inner->dataflow().edges()) {
                auto* source = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
                if (source && source->data() == "b") {
                    ASSERT_EQ(memlet.subset().size(), 1);
                    EXPECT_NE(memlet.subset().at(0)->__str__().find("8*"), std::string::npos)
                        << memlet.subset().at(0)->__str__();
                }
            }
        } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < sequence->size(); i++) {
                check(sequence->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
            check(loop->root());
        }
    };
    check(sdfg.root());
}
