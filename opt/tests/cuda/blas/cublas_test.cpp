#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/targets/cuda/cuda.h"

using namespace sdfg;

TEST(CuBlasTest, DotNodeWithDataTransfers) {
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
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    auto outcome = passes::expansion::expand_single_math_node(builder, block, dot_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(CuBlasTest, DotNodeWithoutDataTransfers) {
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
        block,
        DebugInfo(),
        cuda::ImplementationType_CUDAWithoutTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    auto outcome = passes::expansion::expand_single_math_node(builder, block, dot_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(CuBlasTest, GemmNodeWithDataTransfers) {
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
        cuda::ImplementationType_CUDAWithTransfers,
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

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

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
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    auto map_1 = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(map_1, nullptr);
    EXPECT_EQ(map_1->root().size(), 1);

    auto map_2 = dynamic_cast<structured_control_flow::Map*>(&map_1->root().at(0).first);
    EXPECT_NE(map_2, nullptr);
    EXPECT_EQ(map_2->root().size(), 3);

    auto block_init = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(0).first);
    EXPECT_NE(block_init, nullptr);
    EXPECT_EQ(block_init->dataflow().nodes().size(), 3);
    auto init_tasklet = *block_init->dataflow().tasklets().begin();
    EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);
    EXPECT_EQ(init_tasklet->inputs().at(0), "_in");
    EXPECT_EQ(init_tasklet->output(), "_out");

    auto map_3 = dynamic_cast<structured_control_flow::For*>(&map_2->root().at(1).first);
    EXPECT_NE(map_3, nullptr);
    EXPECT_EQ(map_3->root().size(), 1);

    auto block_fma = dynamic_cast<structured_control_flow::Block*>(&map_3->root().at(0).first);
    EXPECT_NE(block_fma, nullptr);
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 5);

    auto tasklet = *block_fma->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_fma);
    EXPECT_EQ(tasklet->inputs().size(), 3);
    EXPECT_EQ(tasklet->inputs().at(0), "_in1");
    EXPECT_EQ(tasklet->inputs().at(1), "_in2");
    EXPECT_EQ(tasklet->inputs().at(2), "_in3");
    EXPECT_EQ(tasklet->output(), "_out");

    auto block_flush = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(2).first);
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

TEST(CuBlasTest, GemmNodeWithoutDataTransfers) {
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
        cuda::ImplementationType_CUDAWithoutTransfers,
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

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

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
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    auto map_1 = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(map_1, nullptr);
    EXPECT_EQ(map_1->root().size(), 1);

    auto map_2 = dynamic_cast<structured_control_flow::Map*>(&map_1->root().at(0).first);
    EXPECT_NE(map_2, nullptr);
    EXPECT_EQ(map_2->root().size(), 3);

    auto block_init = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(0).first);
    EXPECT_NE(block_init, nullptr);
    EXPECT_EQ(block_init->dataflow().nodes().size(), 3);
    auto init_tasklet = *block_init->dataflow().tasklets().begin();
    EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);
    EXPECT_EQ(init_tasklet->inputs().at(0), "_in");
    EXPECT_EQ(init_tasklet->output(), "_out");

    auto map_3 = dynamic_cast<structured_control_flow::For*>(&map_2->root().at(1).first);
    EXPECT_NE(map_3, nullptr);
    EXPECT_EQ(map_3->root().size(), 1);

    auto block_fma = dynamic_cast<structured_control_flow::Block*>(&map_3->root().at(0).first);
    EXPECT_NE(block_fma, nullptr);
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 5);

    auto tasklet = *block_fma->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_fma);
    EXPECT_EQ(tasklet->inputs().size(), 3);
    EXPECT_EQ(tasklet->inputs().at(0), "_in1");
    EXPECT_EQ(tasklet->inputs().at(1), "_in2");
    EXPECT_EQ(tasklet->inputs().at(2), "_in3");
    EXPECT_EQ(tasklet->output(), "_out");

    auto block_flush = dynamic_cast<structured_control_flow::Block*>(&map_2->root().at(2).first);
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

TEST(CuBlasTest, BatchedGemmNodeWithDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_batch = 4;
    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(dim_batch * dim_i * dim_k));
    types::Array arr_b_type(desc, symbolic::integer(dim_batch * dim_k * dim_j));
    types::Array arr_res_type(desc, symbolic::integer(dim_batch * dim_i * dim_j));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
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
            symbolic::integer(dim_k), // lda
            symbolic::integer(dim_j), // ldb
            symbolic::integer(dim_j), // ldc
            symbolic::integer(dim_i * dim_k), // stride_a
            symbolic::integer(dim_k * dim_j), // stride_b
            symbolic::integer(dim_i * dim_j) // stride_c
        ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, batched_gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, batched_gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder
        .add_computational_memlet(block, dummy_input_node, batched_gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, batched_gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, batched_gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, batched_gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    // After expansion: root should contain a single sequence
    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    // Outermost map is the batch loop
    auto batch_map = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(batch_map, nullptr);
    EXPECT_EQ(batch_map->root().size(), 1);

    // Inside batch loop: map for i dimension
    auto map_i = dynamic_cast<structured_control_flow::Map*>(&batch_map->root().at(0).first);
    EXPECT_NE(map_i, nullptr);
    EXPECT_EQ(map_i->root().size(), 1);

    // Inside i loop: map for j dimension, containing init, k-loop, flush
    auto map_j = dynamic_cast<structured_control_flow::Map*>(&map_i->root().at(0).first);
    EXPECT_NE(map_j, nullptr);
    EXPECT_EQ(map_j->root().size(), 3);

    // Init block
    auto block_init = dynamic_cast<structured_control_flow::Block*>(&map_j->root().at(0).first);
    EXPECT_NE(block_init, nullptr);
    EXPECT_EQ(block_init->dataflow().nodes().size(), 3);
    auto init_tasklet = *block_init->dataflow().tasklets().begin();
    EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);

    // K-loop
    auto for_k = dynamic_cast<structured_control_flow::For*>(&map_j->root().at(1).first);
    EXPECT_NE(for_k, nullptr);
    EXPECT_EQ(for_k->root().size(), 1);

    auto block_fma = dynamic_cast<structured_control_flow::Block*>(&for_k->root().at(0).first);
    EXPECT_NE(block_fma, nullptr);
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 5);
    auto fma_tasklet = *block_fma->dataflow().tasklets().begin();
    EXPECT_EQ(fma_tasklet->code(), data_flow::TaskletCode::fp_fma);

    // Flush block
    auto block_flush = dynamic_cast<structured_control_flow::Block*>(&map_j->root().at(2).first);
    EXPECT_NE(block_flush, nullptr);
    EXPECT_EQ(block_flush->dataflow().nodes().size(), 10);
    auto flush_tasklets = block_flush->dataflow().tasklets();
    EXPECT_EQ(flush_tasklets.size(), 3);
    for (auto* t : flush_tasklets) {
        if (t->code() == data_flow::TaskletCode::fp_add) {
            EXPECT_EQ(t->output(), "_out");
            auto& final_edge = *block_flush->dataflow().out_edges(*t).begin();
            auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
            EXPECT_NE(final_access, nullptr);
            EXPECT_EQ(final_access->data(), c_var_name);
        }
    }
}

TEST(CuBlasTest, BatchedGemmNodeWithoutDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_batch = 4;
    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::integer(dim_batch * dim_i * dim_k));
    types::Array arr_b_type(desc, symbolic::integer(dim_batch * dim_k * dim_j));
    types::Array arr_res_type(desc, symbolic::integer(dim_batch * dim_i * dim_j));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& batched_gemm_node =
        static_cast<math::blas::BatchedGEMMNode&>(builder.add_library_node<math::blas::BatchedGEMMNode>(
            block,
            DebugInfo(),
            cuda::ImplementationType_CUDAWithoutTransfers,
            math::blas::BLAS_Precision::s,
            math::blas::BLAS_Layout::RowMajor,
            math::blas::BLAS_Transpose::No,
            math::blas::BLAS_Transpose::No,
            symbolic::integer(dim_batch),
            symbolic::integer(dim_i),
            symbolic::integer(dim_j),
            symbolic::integer(dim_k),
            symbolic::integer(dim_k), // lda
            symbolic::integer(dim_j), // ldb
            symbolic::integer(dim_j), // ldc
            symbolic::integer(dim_i * dim_k), // stride_a
            symbolic::integer(dim_k * dim_j), // stride_b
            symbolic::integer(dim_i * dim_j) // stride_c
        ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, batched_gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, batched_gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder
        .add_computational_memlet(block, dummy_input_node, batched_gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, batched_gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, batched_gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, batched_gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    // Same structure as WithTransfers - batch loop wrapping i/j/k
    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0).first);
    EXPECT_NE(new_sequence, nullptr);

    auto batch_map = dynamic_cast<structured_control_flow::Map*>(&new_sequence->at(0).first);
    EXPECT_NE(batch_map, nullptr);

    auto map_i = dynamic_cast<structured_control_flow::Map*>(&batch_map->root().at(0).first);
    EXPECT_NE(map_i, nullptr);

    auto map_j = dynamic_cast<structured_control_flow::Map*>(&map_i->root().at(0).first);
    EXPECT_NE(map_j, nullptr);
    EXPECT_EQ(map_j->root().size(), 3);

    auto for_k = dynamic_cast<structured_control_flow::For*>(&map_j->root().at(1).first);
    EXPECT_NE(for_k, nullptr);

    auto block_fma = dynamic_cast<structured_control_flow::Block*>(&for_k->root().at(0).first);
    EXPECT_NE(block_fma, nullptr);
    auto fma_tasklet = *block_fma->dataflow().tasklets().begin();
    EXPECT_EQ(fma_tasklet->code(), data_flow::TaskletCode::fp_fma);

    auto block_flush = dynamic_cast<structured_control_flow::Block*>(&map_j->root().at(2).first);
    EXPECT_NE(block_flush, nullptr);
    EXPECT_EQ(block_flush->dataflow().nodes().size(), 10);
}
