#include <gtest/gtest.h>

#include <cstdint>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(BlasTest, DotNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(2);
    auto stride_b = symbolic::integer(2);

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
        block, DebugInfo(), math::blas::ImplementationType_BLAS, math::blas::BLAS_Precision::d, n, stride_a, stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    auto outcome = passes::expansion::expand_single_math_node(builder, block, dot_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(BlasTest, GemmNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: ixj, A: ixk, B: kxj

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j), // lda
        symbolic::integer(dim_k), // ldb
        symbolic::integer(dim_j) // ldc
    ));

    // Non-special alpha/beta so the full alpha*A*B + beta*C epilogue is generated.
    auto& alpha_node = builder.add_constant(block, "2.0", desc);
    auto& beta_node = builder.add_constant(block, "3.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    dump_sdfg(sdfg, "0.init");

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    dump_sdfg(sdfg, "1.expand");

    builder.subject().validate();

    ASSERT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    ASSERT_NE(new_sequence, nullptr);
    ASSERT_EQ(new_sequence->size(), 2);

    // ---- Init nest: for i, j: C[i,j] = beta * C[i,j] ----
    auto init_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    ASSERT_NE(init_map_i, nullptr);
    EXPECT_EQ(sdfg.type(init_map_i->indvar()->get_name()).primitive_type(), types::PrimitiveType::Int32);
    ASSERT_EQ(init_map_i->root().size(), 1);
    auto init_map_j = dyn_cast<structured_control_flow::Map*>(&init_map_i->root().at(0));
    ASSERT_NE(init_map_j, nullptr);
    EXPECT_EQ(sdfg.type(init_map_j->indvar()->get_name()).primitive_type(), types::PrimitiveType::Int32);
    ASSERT_EQ(init_map_j->root().size(), 1);
    {
        auto block_init = dyn_cast<structured_control_flow::Block*>(&init_map_j->root().at(0));
        ASSERT_NE(block_init, nullptr);
        // beta != 0 and beta != 1: C[i,j] = beta * C[i,j] (c_read, beta, tasklet, c_write)
        ASSERT_EQ(block_init->dataflow().nodes().size(), 4);
        auto init_tasklet = *block_init->dataflow().tasklets().begin();
        ASSERT_EQ(init_tasklet->code(), data_flow::TaskletCode::fp_mul);
        ASSERT_EQ(init_tasklet->output(), "_out");
    }

    // ---- Compute nest: for i, j, k: C[i,j] = alpha * A[i,k] * B[k,j] + C[i,j] ----
    auto comp_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(1));
    ASSERT_NE(comp_map_i, nullptr);
    EXPECT_EQ(sdfg.type(comp_map_i->indvar()->get_name()).primitive_type(), types::PrimitiveType::Int32);
    ASSERT_EQ(comp_map_i->root().size(), 1);
    auto comp_map_j = dyn_cast<structured_control_flow::Map*>(&comp_map_i->root().at(0));
    ASSERT_NE(comp_map_j, nullptr);
    EXPECT_EQ(sdfg.type(comp_map_j->indvar()->get_name()).primitive_type(), types::PrimitiveType::Int32);
    ASSERT_EQ(comp_map_j->root().size(), 1);
    auto comp_for_k = dyn_cast<structured_control_flow::For*>(&comp_map_j->root().at(0));
    ASSERT_NE(comp_for_k, nullptr);
    EXPECT_EQ(sdfg.type(comp_for_k->indvar()->get_name()).primitive_type(), types::PrimitiveType::Int32);
    ASSERT_EQ(comp_for_k->root().size(), 1);
    {
        auto block_fma = dyn_cast<structured_control_flow::Block*>(&comp_for_k->root().at(0));
        ASSERT_NE(block_fma, nullptr);
        // alpha != 1: p = A[i,k] * B[k,j]; C[i,j] = alpha * p + C[i,j]
        // (a, b, c_in, c_out, fma, mul, prod, alpha)
        ASSERT_EQ(block_fma->dataflow().nodes().size(), 8);

        data_flow::Tasklet* fma_tasklet = nullptr;
        for (auto* tasklet : block_fma->dataflow().tasklets()) {
            if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
                fma_tasklet = tasklet;
            }
        }
        ASSERT_NE(fma_tasklet, nullptr);
        ASSERT_EQ(fma_tasklet->inputs().size(), 3);
        ASSERT_EQ(fma_tasklet->inputs().at(0), "_in1");
        ASSERT_EQ(fma_tasklet->inputs().at(1), "_in2");
        ASSERT_EQ(fma_tasklet->inputs().at(2), "_in3");
        ASSERT_EQ(fma_tasklet->output(), "_out");

        // The accumulating store writes back into C.
        auto& final_edge = *block_fma->dataflow().out_edges(*fma_tasklet).begin();
        auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
        ASSERT_NE(final_access, nullptr);
        ASSERT_EQ(final_access->data(), c_var_name);
    }
}


TEST(BlasTest, GemmNode_AlphaOneBetaZero) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: ixj, A: ixk, B: kxj

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j), // lda
        symbolic::integer(dim_k), // ldb
        symbolic::integer(dim_j) // ldc
    ));

    // Special values: alpha == 1 and beta == 0 simplify the epilogue to a plain store C = A*B.
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    EXPECT_NE(new_sequence, nullptr);
    ASSERT_EQ(new_sequence->size(), 2);

    // ---- Init nest: beta == 0 => C[i,j] = 0 (plain store) ----
    auto init_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    ASSERT_NE(init_map_i, nullptr);
    auto init_map_j = dyn_cast<structured_control_flow::Map*>(&init_map_i->root().at(0));
    ASSERT_NE(init_map_j, nullptr);
    auto block_init = dyn_cast<structured_control_flow::Block*>(&init_map_j->root().at(0));
    ASSERT_NE(block_init, nullptr);
    // zero constant, assign tasklet, C write.
    EXPECT_EQ(block_init->dataflow().nodes().size(), 3);
    auto init_tasklets = block_init->dataflow().tasklets();
    EXPECT_EQ(init_tasklets.size(), 1);
    auto* init_tasklet = *init_tasklets.begin();
    EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);

    // ---- Compute nest: alpha == 1 => C[i,j] = A[i,k] * B[k,j] + C[i,j] ----
    auto comp_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(1));
    ASSERT_NE(comp_map_i, nullptr);
    auto comp_map_j = dyn_cast<structured_control_flow::Map*>(&comp_map_i->root().at(0));
    ASSERT_NE(comp_map_j, nullptr);
    auto comp_for_k = dyn_cast<structured_control_flow::For*>(&comp_map_j->root().at(0));
    ASSERT_NE(comp_for_k, nullptr);
    auto block_fma = dyn_cast<structured_control_flow::Block*>(&comp_for_k->root().at(0));
    ASSERT_NE(block_fma, nullptr);
    // alpha == 1: a, b, c_in, c_out, fma (no separate scaling multiply).
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 5);
    auto fma_tasklets = block_fma->dataflow().tasklets();
    EXPECT_EQ(fma_tasklets.size(), 1);
    auto* store_tasklet = *fma_tasklets.begin();
    EXPECT_EQ(store_tasklet->code(), data_flow::TaskletCode::fp_fma);
    auto& final_edge = *block_fma->dataflow().out_edges(*store_tasklet).begin();
    auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
    EXPECT_NE(final_access, nullptr);
    EXPECT_EQ(final_access->data(), c_var_name);
}

TEST(BlasTest, GemmNode_TN) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // trans_a=Trans, trans_b=No
    // A stored as k×m, lda=m; B stored as k×n, ldb=n

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::Trans,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_i), // lda = m (A stored as k×m)
        symbolic::integer(dim_j), // ldb = n (B stored as k×n)
        symbolic::integer(dim_j) // ldc = n
    ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();
}

TEST(BlasTest, GemmNode_NT) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // trans_a=No, trans_b=Trans
    // A stored as m×k, lda=k; B stored as n×k, ldb=k

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::Trans,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k), // lda = k (A stored as m×k)
        symbolic::integer(dim_k), // ldb = k (B stored as n×k)
        symbolic::integer(dim_j) // ldc = n
    ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();
}

TEST(BlasTest, GemmNode_TT) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // trans_a=Trans, trans_b=Trans
    // A stored as k×m, lda=m; B stored as n×k, ldb=k

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::Trans,
        math::blas::BLAS_Transpose::Trans,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_i), // lda = m (A stored as k×m)
        symbolic::integer(dim_k), // ldb = k (B stored as n×k)
        symbolic::integer(dim_j) // ldc = n
    ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();
}

TEST(BlasTest, BatchedGemmNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int batch = 4;
    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: batch×i×j, A: batch×i×k, B: batch×k×j
    int stride_a = dim_i * dim_k;
    int stride_b = dim_k * dim_j;
    int stride_c = dim_i * dim_j;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(batch * stride_a));
    types::Array arr_b_type(desc, symbolic::integer(batch * stride_b));
    types::Array arr_res_type(desc, symbolic::integer(batch * stride_c));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::BatchedGEMMNode&>(builder.add_library_node<math::blas::BatchedGEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(batch),
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k), // lda
        symbolic::integer(dim_j), // ldb
        symbolic::integer(dim_j), // ldc
        symbolic::integer(stride_a),
        symbolic::integer(stride_b),
        symbolic::integer(stride_c)
    ));

    // Non-special alpha/beta so the full alpha*A*B + beta*C epilogue is generated.
    auto& alpha_node = builder.add_constant(block, "2.0", desc);
    auto& beta_node = builder.add_constant(block, "3.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    EXPECT_NE(new_sequence, nullptr);

    // Batch loop (outermost), then i and j maps
    auto batch_loop = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    EXPECT_NE(batch_loop, nullptr);
    auto map_i = dyn_cast<structured_control_flow::Map*>(&batch_loop->root().at(0));
    EXPECT_NE(map_i, nullptr);
    auto map_j = dyn_cast<structured_control_flow::Map*>(&map_i->root().at(0));
    EXPECT_NE(map_j, nullptr);
    EXPECT_EQ(map_j->root().size(), 3);

    // Full epilogue: alpha*sum scaling, beta*C scaling, and the accumulating fp_add store.
    auto block_flush = dyn_cast<structured_control_flow::Block*>(&map_j->root().at(2));
    EXPECT_NE(block_flush, nullptr);
    EXPECT_EQ(block_flush->dataflow().nodes().size(), 10);
    auto flush_tasklets = block_flush->dataflow().tasklets();
    EXPECT_EQ(flush_tasklets.size(), 3);
    for (auto* tasklet : flush_tasklets) {
        if (tasklet->code() == data_flow::TaskletCode::fp_add) {
            EXPECT_EQ(tasklet->output(), "_out");
            auto& final_edge = *block_flush->dataflow().out_edges(*tasklet).begin();
            auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
            EXPECT_NE(final_access, nullptr);
            EXPECT_EQ(final_access->data(), c_var_name);
        }
    }
}

TEST(BlasTest, BatchedGemmNode_AlphaOneBetaZero) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int batch = 4;
    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: batch×i×j, A: batch×i×k, B: batch×k×j
    int stride_a = dim_i * dim_k;
    int stride_b = dim_k * dim_j;
    int stride_c = dim_i * dim_j;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(batch * stride_a));
    types::Array arr_b_type(desc, symbolic::integer(batch * stride_b));
    types::Array arr_res_type(desc, symbolic::integer(batch * stride_c));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::BatchedGEMMNode&>(builder.add_library_node<math::blas::BatchedGEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(batch),
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_k), // lda
        symbolic::integer(dim_j), // ldb
        symbolic::integer(dim_j), // ldc
        symbolic::integer(stride_a),
        symbolic::integer(stride_b),
        symbolic::integer(stride_c)
    ));

    // Special values: alpha == 1 and beta == 0 simplify the epilogue to a plain store C = A*B.
    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    EXPECT_NE(new_sequence, nullptr);

    auto batch_loop = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    EXPECT_NE(batch_loop, nullptr);
    auto map_i = dyn_cast<structured_control_flow::Map*>(&batch_loop->root().at(0));
    EXPECT_NE(map_i, nullptr);
    auto map_j = dyn_cast<structured_control_flow::Map*>(&map_i->root().at(0));
    EXPECT_NE(map_j, nullptr);
    EXPECT_EQ(map_j->root().size(), 3);

    // The flush block must be a plain store: one assign tasklet, no scaling (fp_mul) and no accumulation (fp_add).
    auto block_flush = dyn_cast<structured_control_flow::Block*>(&map_j->root().at(2));
    EXPECT_NE(block_flush, nullptr);
    EXPECT_EQ(block_flush->dataflow().nodes().size(), 3);
    auto flush_tasklets = block_flush->dataflow().tasklets();
    EXPECT_EQ(flush_tasklets.size(), 1);
    auto* store_tasklet = *flush_tasklets.begin();
    EXPECT_EQ(store_tasklet->code(), data_flow::TaskletCode::assign);
    auto& final_edge = *block_flush->dataflow().out_edges(*store_tasklet).begin();
    auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
    EXPECT_NE(final_access, nullptr);
    EXPECT_EQ(final_access->data(), c_var_name);
}

// ---------------------------------------------------------------------------
// Parameterized GEMM expansion test focused on the induction-variable types.
//
// The expansion picks each loop's index type from its dimension bound via
// types::get_primitive_type_to_hold_expression: bounds that fit in INT32_MAX
// stay Int32, everything larger (beyond Int32 *and* beyond UInt32) becomes
// Int64. The i/j/k induction variables come from m/n/k respectively, so mixing
// the dimension magnitudes lets each index resolve to a different width.
// ---------------------------------------------------------------------------

namespace {

// Reference thresholds for readable parameter values.
constexpr int64_t kInt32Max = INT32_MAX; // 2^31 - 1
constexpr int64_t kUInt32Max = UINT32_MAX; // 2^32 - 1
constexpr int64_t kAboveInt32 = kInt32Max + 1; // 2^31, no longer fits Int32
constexpr int64_t kAboveUInt32 = kUInt32Max + 1; // 2^32, beyond UInt32 too

struct GemmIndvarTypeParams {
    std::string label;
    int64_t dim_i; // -> i induction variable (from m)
    int64_t dim_j; // -> j induction variable (from n)
    int64_t dim_k; // -> k induction variable (from k)
    types::PrimitiveType expected_i;
    types::PrimitiveType expected_j;
    types::PrimitiveType expected_k;
};

class GemmNodeIndvarTypeTest : public ::testing::TestWithParam<GemmIndvarTypeParams> {};

TEST_P(GemmNodeIndvarTypeTest, ExpandedIndvarTypesMatchDimensionMagnitude) {
    const auto& p = GetParam();

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto m = symbolic::integer(p.dim_i);
    auto n = symbolic::integer(p.dim_j);
    auto k = symbolic::integer(p.dim_k);

    // res: ixj, A: ixk, B: kxj. Sizes are symbolic (arbitrary precision), so
    // huge dimensions do not overflow the container descriptors.
    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(k, m));
    types::Array arr_b_type(desc, symbolic::mul(n, k));
    types::Array arr_res_type(desc, symbolic::mul(n, m));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto& dummy_input_node = builder.add_access(block, "output");
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        m,
        n,
        k,
        n, // lda
        k, // ldb
        n // ldc
    ));

    // Non-special alpha/beta so both the init and compute nests are generated.
    auto& alpha_node = builder.add_constant(block, "2.0", desc);
    auto& beta_node = builder.add_constant(block, "3.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    ASSERT_TRUE(outcome.expanded);
    dump_sdfg(builder.subject(), "1.expanded");
    ASSERT_TRUE(outcome.block_removed);
    builder.subject().validate();

    ASSERT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    ASSERT_NE(new_sequence, nullptr);
    // beta != 1 => [init nest, compute nest]; the compute nest carries all three
    // (i, j, k) induction variables, so it is enough to inspect it.
    ASSERT_EQ(new_sequence->size(), 2);

    auto comp_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(1));
    ASSERT_NE(comp_map_i, nullptr);
    auto comp_map_j = dyn_cast<structured_control_flow::Map*>(&comp_map_i->root().at(0));
    ASSERT_NE(comp_map_j, nullptr);
    auto comp_for_k = dyn_cast<structured_control_flow::For*>(&comp_map_j->root().at(0));
    ASSERT_NE(comp_for_k, nullptr);

    EXPECT_EQ(sdfg.type(comp_map_i->indvar()->get_name()).primitive_type(), p.expected_i);
    EXPECT_EQ(sdfg.type(comp_map_j->indvar()->get_name()).primitive_type(), p.expected_j);
    EXPECT_EQ(sdfg.type(comp_for_k->indvar()->get_name()).primitive_type(), p.expected_k);
}

INSTANTIATE_TEST_SUITE_P(
    BlasTest,
    GemmNodeIndvarTypeTest,
    ::testing::Values(
        // All dimensions small: every index fits in Int32.
        GemmIndvarTypeParams{
            "AllInt32", 10, 20, 30, types::PrimitiveType::Int32, types::PrimitiveType::Int32, types::PrimitiveType::Int32
        },
        // One dimension just past Int32 (still within UInt32) -> that index is
        // promoted to Int64, the others stay Int32. Magnitudes are mixed.
        GemmIndvarTypeParams{
            "MixedAboveInt32",
            kAboveInt32,
            20,
            30,
            types::PrimitiveType::Int64,
            types::PrimitiveType::Int32,
            types::PrimitiveType::Int32
        },
        // Mix of "> Int32" and "> UInt32": both promote to Int64, small stays Int32.
        GemmIndvarTypeParams{
            "MixedAboveUInt32",
            kAboveUInt32,
            kAboveInt32,
            30,
            types::PrimitiveType::Int64,
            types::PrimitiveType::Int64,
            types::PrimitiveType::Int32
        },
        // Fully mixed across the three regimes: small / > Int32 / > UInt32.
        GemmIndvarTypeParams{
            "MixedAllRegimes",
            10,
            kAboveInt32,
            kAboveUInt32,
            types::PrimitiveType::Int32,
            types::PrimitiveType::Int64,
            types::PrimitiveType::Int64
        }
    ),
    [](const ::testing::TestParamInfo<GemmIndvarTypeParams>& info) { return info.param.label; }
);

} // namespace
