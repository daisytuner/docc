#include "sdfg/einsum/einsum_replacer.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/einsum/einsum_detection.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(Einsum2MatMulTest, MatMul) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Loop induction variables and (external) bounds.
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("K", sym_desc, true);

    // 2D row-major matrices so that a TensorLayout can be inferred for each operand.
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array a_type(types::Array(base_desc, symbolic::symbol("K")), symbolic::symbol("M"));
    types::Array b_type(types::Array(base_desc, symbolic::symbol("N")), symbolic::symbol("K"));
    types::Array c_type(types::Array(base_desc, symbolic::symbol("N")), symbolic::symbol("M"));
    builder.add_container("A", a_type, true);
    builder.add_container("B", b_type, true);
    builder.add_container("C", c_type, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");

    // Loop nest: for i < M, j < N, k < K
    auto& map_i =
        builder.add_map(root, i, symbolic::Lt(i, M), zero, symbolic::add(i, one), ScheduleType_Sequential::create());
    auto& map_j =
        builder
            .add_map(map_i.root(), j, symbolic::Lt(j, N), zero, symbolic::add(j, one), ScheduleType_Sequential::create());
    auto& for_k = builder.add_for(map_j.root(), k, symbolic::Lt(k, K), zero, symbolic::add(k, one));

    // Canonical GEMM core (no alpha): C[i,j] += A[i,k] * B[k,j]
    auto& block = builder.add_block(for_k.root());
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {i, k});
    builder.add_computational_memlet(block, B, tasklet, "_in2", {k, j});
    builder.add_computational_memlet(block, C1, tasklet, "_in3", {i, j});
    builder.add_computational_memlet(block, tasklet, "_out", C2, {i, j});

    dump_sdfg(builder.subject(), "0.init");

    // Detect the single einsum cluster spanning all three loops.
    einsum::EinsumDetection einsum_detection;
    ASSERT_EQ(einsum_detection.run(root), 1);
    auto& cluster = *einsum_detection.einsums().at(0);

    // Replace it with a MatMulNode.
    einsum::Einsum2MatMul replacer;
    ASSERT_TRUE(replacer.can_be_applied(cluster));
    auto outcome = einsum::replace_einsum_cluster(builder, cluster, replacer);
    EXPECT_TRUE(outcome.applied);

    dump_sdfg(builder.subject(), "1.replaced");

    // The original loop nest is gone, replaced by a sequence holding a single MatMul block.
    ASSERT_EQ(root.size(), 1);
    auto* sequence = dyn_cast<structured_control_flow::Sequence*>(&root.at(0));
    ASSERT_TRUE(sequence);
    ASSERT_EQ(sequence->size(), 1);

    auto* matmul_block = dyn_cast<structured_control_flow::Block*>(&sequence->at(0));
    ASSERT_TRUE(matmul_block);
    EXPECT_EQ(matmul_block->dataflow().tasklets().size(), 0);
    ASSERT_EQ(matmul_block->dataflow().library_nodes().size(), 1);

    auto* matmul = dynamic_cast<math::tensor::MatMulNode*>(*matmul_block->dataflow().library_nodes().begin());
    EXPECT_TRUE(matmul);
    EXPECT_EQ(matmul_block->dataflow().data_nodes().size(), 3);
}
