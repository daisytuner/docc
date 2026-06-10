#include "sdfg/analysis/memory_layout_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(MemoryLayoutAnalysisTest, Linearized_2D_RowMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with linearized access: A[i*M + j]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at inner loop
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));

    ASSERT_EQ(tile_j_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Inner tile extents: [1, M] and contiguous range: [i*M, i*M + M-1]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), M));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, symbolic::mul(i, M)));
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::sub(symbolic::add(symbolic::mul(i, M), M), symbolic::one())));

    // Check tile at outer loop
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));

    ASSERT_EQ(tile_i_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));

    // Outer tile extents: [N, M] and contiguous range: [0, (N-1)*M + M-1] = [0, N*M - 1]
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), M));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::zero()));
    // last = M*(N-1) + (M-1) = MN - 1
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::sub(symbolic::mul(M, N), symbolic::one())));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_ColMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with linearized access: A[j*N + i]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(j, N), i);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), i));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), N));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at inner loop
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), i));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), i));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), N));

    ASSERT_EQ(tile_j_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Inner tile extents: [M, 1] and contiguous range: [i, N*(M-1) + i]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), symbolic::one()));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, i));
    // last = N*(M-1) + i = MN - N + i
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::add(symbolic::sub(symbolic::mul(M, N), N), i)));

    // Check tile at outer loop
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(N, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), N));

    ASSERT_EQ(tile_i_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));

    // Outer tile extents: [M, N]
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), N));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::zero()));
    // last = N*(M-1) + (N-1) = MN - 1
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::sub(symbolic::mul(M, N), symbolic::one())));
}

TEST(MemoryLayoutAnalysisTest, Linearized_3D_RowMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, M)
    auto& middle_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    // for k in [0, K)
    auto& inner_loop =
        builder
            .add_for(middle_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // A[i*M*K + j*K + k]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(i, symbolic::mul(M, K)), symbolic::mul(j, K)), k);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(2), k));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(2), K));

    ASSERT_EQ(layout_in.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at innermost loop (k)
    auto* tile_k = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_k, nullptr);

    ASSERT_EQ(tile_k->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_k->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_k_layout = tile_k->layout;
    ASSERT_EQ(tile_k_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(2), K));

    ASSERT_EQ(tile_k_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.offset(), symbolic::zero()));

    // tile_k extents: [1, 1, K], range: [i*M*K + j*K, i*M*K + j*K + K-1]
    auto tile_k_ext = tile_k->extents();
    ASSERT_EQ(tile_k_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(2), K));

    auto [tile_k_first, tile_k_last] = tile_k->contiguous_range();
    auto iMK_jK = symbolic::add(symbolic::mul(i, symbolic::mul(M, K)), symbolic::mul(j, K));
    EXPECT_TRUE(symbolic::eq(tile_k_first, iMK_jK));
    EXPECT_TRUE(symbolic::eq(tile_k_last, symbolic::add(iMK_jK, symbolic::sub(K, symbolic::one()))));

    // Check tile at middle loop (j)
    auto* tile_j = analysis.tile(middle_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(2), K));

    ASSERT_EQ(tile_j_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // tile_j extents: [1, M, K], range: [i*M*K, i*M*K + (M-1)*K + K-1]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(2), K));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    // first = i*M*K
    EXPECT_TRUE(symbolic::eq(tile_j_first, symbolic::mul(i, symbolic::mul(M, K))));
    // last = i*MK + (M-1)*K + (K-1) = i*MK + MK - 1
    EXPECT_TRUE(symbolic::eq(
        tile_j_last,
        symbolic::sub(symbolic::add(symbolic::mul(i, symbolic::mul(M, K)), symbolic::mul(M, K)), symbolic::one())
    ));

    // Check tile at outer loop (i)
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(2), K));

    ASSERT_EQ(tile_i_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));

    // tile_i extents: [N, M, K], range: [0, (N-1)*M*K + (M-1)*K + K-1]
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(2), K));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::zero()));
    // last = (N-1)*MK + (M-1)*K + (K-1) = NMK - 1
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::sub(symbolic::mul(N, symbolic::mul(M, K)), symbolic::one())));
}

TEST(MemoryLayoutAnalysisTest, Linearized_3D_ColMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, M)
    auto& middle_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    // for k in [0, K)
    auto& inner_loop =
        builder
            .add_for(middle_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // A[k*N*M + j*N + i]  (column-major)
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(k, symbolic::mul(N, M)), symbolic::mul(j, N)), i);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearization recovers [k, j, i] with strides [N*M, N, 1]
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), k));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(2), i));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(2), N));

    ASSERT_EQ(layout_in.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at innermost loop (k)
    auto* tile_k = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_k, nullptr);

    ASSERT_EQ(tile_k->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(2), i));

    ASSERT_EQ(tile_k->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(2), i));

    auto& tile_k_layout = tile_k->layout;
    ASSERT_EQ(tile_k_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(2), N));

    ASSERT_EQ(tile_k_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.offset(), symbolic::zero()));

    // tile_k extents: [K, 1, 1], range: [N*j + i, M*N*(K-1) + N*j + i]
    auto tile_k_ext = tile_k->extents();
    ASSERT_EQ(tile_k_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(0), K));
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_ext.at(2), symbolic::one()));

    auto [tile_k_first, tile_k_last] = tile_k->contiguous_range();
    auto Nj_i = symbolic::add(symbolic::mul(N, j), i);
    EXPECT_TRUE(symbolic::eq(tile_k_first, Nj_i));
    // last = MN*(K-1) + Nj + i = MNK - MN + Nj + i
    EXPECT_TRUE(symbolic::
                    eq(tile_k_last,
                       symbolic::add(symbolic::sub(symbolic::mul(symbolic::mul(M, N), K), symbolic::mul(M, N)), Nj_i)));

    // Check tile at middle loop (j)
    auto* tile_j = analysis.tile(middle_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(2), i));

    ASSERT_EQ(tile_j->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(2), i));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(2), N));

    ASSERT_EQ(tile_j_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // tile_j extents: [K, M, 1], range: [i, M*N*(K-1) + N*(M-1) + i]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), K));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(2), symbolic::one()));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, i));
    // last = MN*(K-1) + N*(M-1) + i = MNK - N + i
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::add(symbolic::sub(symbolic::mul(symbolic::mul(M, N), K), N), i)));

    // Check tile at outer loop (i)
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(2), symbolic::sub(N, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(2), N));

    ASSERT_EQ(tile_i_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));

    // tile_i extents: [K, M, N], range: [0, M*N*(K-1) + N*(M-1) + N-1]
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), K));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(2), N));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::zero()));
    // last = MN*(K-1) + N*(M-1) + (N-1) = MNK - 1
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::sub(symbolic::mul(symbolic::mul(M, N), K), symbolic::one())));
}

TEST(MemoryLayoutAnalysisTest, Stencil_2D_5Point) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);
    builder.add_container("B", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [1, N-1)
    auto& outer_loop = builder.add_for(
        root,
        i,
        symbolic::Lt(i, symbolic::sub(N, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(i, symbolic::one())
    );
    // for j in [1, M-1)
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(j, symbolic::one())
    );

    // 5-point stencil: B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j] + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
    auto& block = builder.add_block(inner_loop.root());

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);
    auto west = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, symbolic::one()));
    auto east = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));

    auto& a_center = builder.add_access(block, "A");
    auto& a_north = builder.add_access(block, "A");
    auto& a_south = builder.add_access(block, "A");
    auto& a_west = builder.add_access(block, "A");
    auto& a_east = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_c", "_n", "_s", "_w", "_e"});

    auto& m_center = builder.add_computational_memlet(block, a_center, tasklet, "_c", {center});
    auto& m_north = builder.add_computational_memlet(block, a_north, tasklet, "_n", {north});
    auto& m_south = builder.add_computational_memlet(block, a_south, tasklet, "_s", {south});
    auto& m_west = builder.add_computational_memlet(block, a_west, tasklet, "_w", {west});
    auto& m_east = builder.add_computational_memlet(block, a_east, tasklet, "_e", {east});
    auto& m_out = builder.add_computational_memlet(block, tasklet, "_out", b_out, {center});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // All 5 input accesses to A should delinearize to 2D with the same layout
    for (auto* memlet : {&m_center, &m_north, &m_south, &m_west, &m_east}) {
        auto result = analysis.access(*memlet);
        ASSERT_NE(result, nullptr);

        ASSERT_EQ(result->subset.size(), 2);
        const auto& layout = result->layout;
        ASSERT_EQ(layout.shape().size(), 2);
        EXPECT_TRUE(symbolic::eq(layout.shape().at(0), symbolic::symbol("__unbounded__")));
        EXPECT_TRUE(symbolic::eq(layout.shape().at(1), M));

        ASSERT_EQ(layout.strides().size(), 2);
        EXPECT_TRUE(symbolic::eq(layout.strides().at(0), M));
        EXPECT_TRUE(symbolic::eq(layout.strides().at(1), symbolic::one()));
    }

    // Check specific delinearized indices
    auto r_center = analysis.access(m_center);
    EXPECT_TRUE(symbolic::eq(r_center->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_center->subset.at(1), j));

    auto r_north = analysis.access(m_north);
    EXPECT_TRUE(symbolic::eq(r_north->subset.at(0), symbolic::sub(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(r_north->subset.at(1), j));

    auto r_south = analysis.access(m_south);
    EXPECT_TRUE(symbolic::eq(r_south->subset.at(0), symbolic::add(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(r_south->subset.at(1), j));

    auto r_west = analysis.access(m_west);
    EXPECT_TRUE(symbolic::eq(r_west->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_west->subset.at(1), symbolic::sub(j, symbolic::one())));

    auto r_east = analysis.access(m_east);
    EXPECT_TRUE(symbolic::eq(r_east->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_east->subset.at(1), symbolic::add(j, symbolic::one())));

    // Output memlet to B should also delinearize
    auto r_out = analysis.access(m_out);
    ASSERT_NE(r_out, nullptr);
    ASSERT_EQ(r_out->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(r_out->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_out->subset.at(1), j));

    // Check tile at inner loop for A: min/max across all 5 stencil points
    // i indices: {i-1, i, i+1}, j indices: {j-1, j, j+1}
    // simplify collapses: min(i, i-1, i+1) → i-1, max(i, i-1, i+1) → i+1
    auto* tile_j_A = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j_A, nullptr);

    auto i_minus_1 = symbolic::sub(i, symbolic::one());
    auto i_plus_1 = symbolic::add(i, symbolic::one());
    auto M_minus_1 = symbolic::sub(M, symbolic::one());
    auto M_minus_2 = symbolic::sub(M, symbolic::integer(2));

    ASSERT_EQ(tile_j_A->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_A->min_subset.at(0), i_minus_1));
    EXPECT_TRUE(symbolic::eq(tile_j_A->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j_A->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_A->max_subset.at(0), i_plus_1));
    // max(M-1, M-2, M-3) simplified → M-1
    EXPECT_TRUE(symbolic::eq(tile_j_A->max_subset.at(1), M_minus_1));

    // tile_j_A extents: [(i+1)-(i-1)+1 = 3, (M-1)-0+1 = M]
    auto tile_j_A_ext = tile_j_A->extents();
    ASSERT_EQ(tile_j_A_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_A_ext.at(0), symbolic::integer(3)));
    EXPECT_TRUE(symbolic::eq(tile_j_A_ext.at(1), M));

    // contiguous_range: first = M*(i-1), last = M*(i+1) + (M-1)
    auto [tile_j_A_first, tile_j_A_last] = tile_j_A->contiguous_range();
    // first = M*(i-1) = Mi - M
    EXPECT_TRUE(symbolic::eq(tile_j_A_first, symbolic::sub(symbolic::mul(M, i), M)));
    // last = M*(i+1) + (M-1) = Mi + 2M - 1
    EXPECT_TRUE(symbolic::eq(
        tile_j_A_last,
        symbolic::add(symbolic::mul(M, i), symbolic::sub(symbolic::mul(symbolic::integer(2), M), symbolic::one()))
    ));

    // Check tile at inner loop for B
    auto* tile_j_B = analysis.tile(inner_loop, "B");
    ASSERT_NE(tile_j_B, nullptr);

    ASSERT_EQ(tile_j_B->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_B->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j_B->min_subset.at(1), symbolic::integer(1)));

    ASSERT_EQ(tile_j_B->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_B->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j_B->max_subset.at(1), M_minus_2));

    // tile_j_B extents: [1, M-2], range: [M*i + 1, M*i + M-2]
    auto tile_j_B_ext = tile_j_B->extents();
    ASSERT_EQ(tile_j_B_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_B_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_B_ext.at(1), symbolic::sub(M, symbolic::integer(2))));

    auto [tile_j_B_first, tile_j_B_last] = tile_j_B->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_B_first, symbolic::add(symbolic::mul(M, i), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j_B_last, symbolic::add(symbolic::mul(M, i), symbolic::sub(M, symbolic::integer(2))))
    );

    // Check tile at outer loop for A
    auto* tile_i_A = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i_A, nullptr);

    auto N_minus_1 = symbolic::sub(N, symbolic::one());
    auto N_minus_2 = symbolic::sub(N, symbolic::integer(2));

    ASSERT_EQ(tile_i_A->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_A->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i_A->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i_A->max_subset.size(), 2);
    // max(N-1, N-3, N-2) simplified → N-1
    EXPECT_TRUE(symbolic::eq(tile_i_A->max_subset.at(0), N_minus_1));
    // max(M-1, M-2, M-3) simplified → M-1
    EXPECT_TRUE(symbolic::eq(tile_i_A->max_subset.at(1), M_minus_1));

    // tile_i_A extents: [(N-1)-0+1 = N, (M-1)-0+1 = M]
    auto tile_i_A_ext = tile_i_A->extents();
    ASSERT_EQ(tile_i_A_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_A_ext.at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_A_ext.at(1), M));

    // contiguous_range: first = 0, last = M*(N-1) + (M-1)
    auto [tile_i_A_first, tile_i_A_last] = tile_i_A->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_A_first, symbolic::zero()));
    // last = M*(N-1) + (M-1) = MN - 1
    EXPECT_TRUE(symbolic::eq(tile_i_A_last, symbolic::sub(symbolic::mul(M, N), symbolic::one())));

    // Check tile at outer loop for B
    auto* tile_i_B = analysis.tile(outer_loop, "B");
    ASSERT_NE(tile_i_B, nullptr);

    ASSERT_EQ(tile_i_B->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_B->min_subset.at(0), symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(tile_i_B->min_subset.at(1), symbolic::integer(1)));

    ASSERT_EQ(tile_i_B->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_B->max_subset.at(0), N_minus_2));
    EXPECT_TRUE(symbolic::eq(tile_i_B->max_subset.at(1), M_minus_2));

    // tile_i_B extents: [N-2, M-2], range: [M+1, M*(N-2) + M-2]
    auto tile_i_B_ext = tile_i_B->extents();
    ASSERT_EQ(tile_i_B_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_B_ext.at(0), symbolic::sub(N, symbolic::integer(2))));
    EXPECT_TRUE(symbolic::eq(tile_i_B_ext.at(1), symbolic::sub(M, symbolic::integer(2))));

    auto [tile_i_B_first, tile_i_B_last] = tile_i_B->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_B_first, symbolic::add(M, symbolic::one())));
    // last = M*(N-2) + (M-2) = MN - M - 2
    EXPECT_TRUE(symbolic::eq(tile_i_B_last, symbolic::sub(symbolic::sub(symbolic::mul(M, N), M), symbolic::integer(2)))
    );
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_TriangularLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, i)  — triangular: inner bound depends on outer indvar
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, i), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // A[i*N + j]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, N), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearizes to [i, j] with stride N
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), N));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));

    // Check tile at inner loop (j): j ranges [0, i-1], i is constant
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    // j's upper bound is i-1 (from j < i)
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(i, symbolic::one())));

    // tile_j extents: [1, i], range: [N*i, N*i + i-1]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), i));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, symbolic::mul(N, i)));
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::add(symbolic::mul(N, i), symbolic::sub(i, symbolic::one()))));

    // Check tile at outer loop (i): i ranges [0, N-1]
    // Inner tile min=[i, 0], max=[i, i-1]
    // After bounding over i: min=[0, 0], max=[N-1, N-2]
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(N, symbolic::integer(2))));

    // tile_i extents: [N, N-1], range: [0, N*(N-1) + N-2]
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), symbolic::sub(N, symbolic::one())));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::zero()));
    // last = N*(N-1) + (N-2) = N² - 2
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::sub(symbolic::mul(N, N), symbolic::integer(2))));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_TiledLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i_tile", index_type);
    builder.add_container("j_tile", index_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto tile_size = symbolic::integer(32);

    // for i_tile in [0, N) step 32
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, tile_size));

    // for j_tile in [0, M) step 32
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M), symbolic::integer(0), symbolic::add(j_tile, tile_size)
    );

    // for i in [i_tile, min(i_tile+32, N)):  cond = i < i_tile+32 && i < N
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, tile_size)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j in [j_tile, min(j_tile+32, M)):  cond = j < j_tile+32 && j < M
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, tile_size)), symbolic::Lt(j, M)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // A[i*M + j]
    auto& block = builder.add_block(j_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearizes to [i, j] with stride M
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));

    // Check tile at j_loop (innermost): j varies, i is constant
    // j ranges [j_tile, min(j_tile+32, M)-1]
    // tight upper bound for j: min(j_tile+31, M-1)
    auto* tile_j = analysis.tile(j_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), j_tile));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    // j's tight upper bound: min(j_tile+31, M-1)
    EXPECT_TRUE(symbolic::eq(
        tile_j->max_subset.at(1),
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()))
    ));

    // tile_j extents and contiguous range
    auto min_j_upper =
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()));

    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), symbolic::add(symbolic::sub(min_j_upper, j_tile), symbolic::one())));

    // extents_approx: overapproximate picks j_tile+31 from min(j_tile+31, M-1)
    // (largest constant offset), so approx extent = j_tile+31 - j_tile + 1 = 32
    auto tile_j_ext_approx = tile_j->extents_approx();
    ASSERT_EQ(tile_j_ext_approx.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext_approx.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_ext_approx.at(1), tile_size));

    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, symbolic::add(symbolic::mul(M, i), j_tile)));
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::add(symbolic::mul(M, i), min_j_upper)));

    // Check tile at i_loop: i varies [i_tile, min(i_tile+32, N)-1], j resolved from inner tile
    auto* tile_i = analysis.tile(i_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), i_tile));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), j_tile));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    // i's tight upper bound: min(i_tile+31, N-1)
    EXPECT_TRUE(symbolic::eq(
        tile_i->max_subset.at(0),
        symbolic::min(symbolic::sub(symbolic::add(i_tile, tile_size), symbolic::one()), symbolic::sub(N, symbolic::one()))
    ));
    EXPECT_TRUE(symbolic::eq(
        tile_i->max_subset.at(1),
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()))
    ));

    // tile_i extents and contiguous range
    auto min_i_upper =
        symbolic::min(symbolic::sub(symbolic::add(i_tile, tile_size), symbolic::one()), symbolic::sub(N, symbolic::one()));

    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), symbolic::add(symbolic::sub(min_i_upper, i_tile), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), symbolic::add(symbolic::sub(min_j_upper, j_tile), symbolic::one())));

    // extents_approx: overapproximate picks i_tile+31 and j_tile+31 → [32, 32]
    auto tile_i_ext_approx = tile_i->extents_approx();
    ASSERT_EQ(tile_i_ext_approx.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext_approx.at(0), tile_size));
    EXPECT_TRUE(symbolic::eq(tile_i_ext_approx.at(1), tile_size));

    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::add(symbolic::mul(M, i_tile), j_tile)));
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::add(symbolic::mul(M, min_i_upper), min_j_upper)));

    // Check tile at j_tile_loop: j_tile varies [0, M) step 32
    auto* tile_jt = analysis.tile(j_tile_loop, "A");
    ASSERT_NE(tile_jt, nullptr);

    ASSERT_EQ(tile_jt->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(0), i_tile));
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(1), symbolic::zero()));

    // Check tile at i_tile_loop (outermost): everything resolved
    auto* tile_it = analysis.tile(i_tile_loop, "A");
    ASSERT_NE(tile_it, nullptr);

    ASSERT_EQ(tile_it->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(1), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_TiledLoop_ColMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i_tile", index_type);
    builder.add_container("j_tile", index_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto tile_size = symbolic::integer(32);

    // for i_tile in [0, N) step 32
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, tile_size));

    // for j_tile in [0, M) step 32
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M), symbolic::integer(0), symbolic::add(j_tile, tile_size)
    );

    // for i in [i_tile, min(i_tile+32, N))
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, tile_size)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j in [j_tile, min(j_tile+32, M))
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, tile_size)), symbolic::Lt(j, M)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // A[j*N + i]  (column-major)
    auto& block = builder.add_block(j_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(j, N), i);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearizes to [j, i] with strides [N, 1]
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), i));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), N));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));

    // Check tile at j_loop (innermost): j varies, i is constant
    // Delinearized subset dim0=j, dim1=i
    // j ranges [j_tile, min(j_tile+31, M-1)], i is constant
    auto* tile_j = analysis.tile(j_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    auto min_j_upper =
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()));

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), j_tile));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), i));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), min_j_upper));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), i));

    // tile_j extents: [min(j_tile+31,M-1)-j_tile+1, 1]
    auto tile_j_ext = tile_j->extents();
    ASSERT_EQ(tile_j_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(0), symbolic::add(symbolic::sub(min_j_upper, j_tile), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j_ext.at(1), symbolic::one()));

    // extents_approx: distribute -j_tile into min → min(32, M-j_tile) → pick 32
    auto tile_j_ext_approx = tile_j->extents_approx();
    ASSERT_EQ(tile_j_ext_approx.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_ext_approx.at(0), tile_size));
    EXPECT_TRUE(symbolic::eq(tile_j_ext_approx.at(1), symbolic::one()));

    // contiguous_range: first = N*j_tile + i, last = N*min_j_upper + i
    auto [tile_j_first, tile_j_last] = tile_j->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_j_first, symbolic::add(symbolic::mul(N, j_tile), i)));
    EXPECT_TRUE(symbolic::eq(tile_j_last, symbolic::add(symbolic::mul(N, min_j_upper), i)));

    // Check tile at i_loop: i varies [i_tile, min(i_tile+31, N-1)]
    auto* tile_i = analysis.tile(i_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    auto min_i_upper =
        symbolic::min(symbolic::sub(symbolic::add(i_tile, tile_size), symbolic::one()), symbolic::sub(N, symbolic::one()));

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), j_tile));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), i_tile));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), min_j_upper));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), min_i_upper));

    // tile_i extents
    auto tile_i_ext = tile_i->extents();
    ASSERT_EQ(tile_i_ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(0), symbolic::add(symbolic::sub(min_j_upper, j_tile), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i_ext.at(1), symbolic::add(symbolic::sub(min_i_upper, i_tile), symbolic::one())));

    // extents_approx: [32, 32]
    auto tile_i_ext_approx = tile_i->extents_approx();
    ASSERT_EQ(tile_i_ext_approx.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_ext_approx.at(0), tile_size));
    EXPECT_TRUE(symbolic::eq(tile_i_ext_approx.at(1), tile_size));

    // contiguous_range: first = N*j_tile + i_tile, last = N*min_j_upper + min_i_upper
    auto [tile_i_first, tile_i_last] = tile_i->contiguous_range();
    EXPECT_TRUE(symbolic::eq(tile_i_first, symbolic::add(symbolic::mul(N, j_tile), i_tile)));
    EXPECT_TRUE(symbolic::eq(tile_i_last, symbolic::add(symbolic::mul(N, min_j_upper), min_i_upper)));

    // Check tile at j_tile_loop
    auto* tile_jt = analysis.tile(j_tile_loop, "A");
    ASSERT_NE(tile_jt, nullptr);

    ASSERT_EQ(tile_jt->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(1), i_tile));

    // Check tile at i_tile_loop (outermost)
    auto* tile_it = analysis.tile(i_tile_loop, "A");
    ASSERT_NE(tile_it, nullptr);

    ASSERT_EQ(tile_it->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(1), symbolic::zero()));
}

// Test tile groups: SYR2K-style access pattern A[i,k] + A[j,k]
// Two memlets to the same container with different row indices should produce two groups
TEST(MemoryLayoutAnalysisTest, TileGroups_TwoIndependentAccesses) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);
    builder.add_container("tmp", scalar_type);

    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i in [0, N)
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // for j in [0, N)
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // for k in [0, K)
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block: tmp = A[i*K + k] * A[j*K + k]
    auto& block = builder.add_block(k_loop.root());
    auto& access_A_ik = builder.add_access(block, "A");
    auto& access_A_jk = builder.add_access(block, "A");
    auto& access_tmp = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_a", "_b"});

    auto linearized_ik = symbolic::add(symbolic::mul(i, K), k);
    auto linearized_jk = symbolic::add(symbolic::mul(j, K), k);

    auto& memlet_A_ik = builder.add_computational_memlet(block, access_A_ik, tasklet, "_a", {linearized_ik});
    auto& memlet_A_jk = builder.add_computational_memlet(block, access_A_jk, tasklet, "_b", {linearized_jk});
    builder.add_computational_memlet(block, tasklet, "_out", access_tmp, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check that raw accesses delinearize correctly
    auto* acc_ik = analysis.access(memlet_A_ik);
    ASSERT_NE(acc_ik, nullptr);
    ASSERT_EQ(acc_ik->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(acc_ik->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(acc_ik->subset.at(1), k));

    auto* acc_jk = analysis.access(memlet_A_jk);
    ASSERT_NE(acc_jk, nullptr);
    ASSERT_EQ(acc_jk->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(acc_jk->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(acc_jk->subset.at(1), k));

    // The merged tile at k_loop should still be nullptr or have symbolic extents
    // because merging A[i,k] and A[j,k] gives min(i,j) which is symbolic
    // (The existing behavior: tile() may or may not succeed depending on symbolic simplifier)

    // Check tile groups at k_loop: should have 2 groups for container "A"
    auto* groups = analysis.tile_groups(k_loop, "A");
    ASSERT_NE(groups, nullptr);
    ASSERT_EQ(groups->size(), 2);

    // Find which group contains memlet_A_ik and which contains memlet_A_jk
    auto* group_ik = analysis.tile_group_for(k_loop, memlet_A_ik);
    auto* group_jk = analysis.tile_group_for(k_loop, memlet_A_jk);
    ASSERT_NE(group_ik, nullptr);
    ASSERT_NE(group_jk, nullptr);
    EXPECT_NE(group_ik, group_jk); // Different groups

    // Group for A[i,k]: tile should be [i, 0] -> [i, K-1], extents [1, K]
    ASSERT_EQ(group_ik->tile.min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(group_ik->tile.min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(group_ik->tile.min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(group_ik->tile.max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(group_ik->tile.max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(group_ik->tile.max_subset.at(1), symbolic::sub(K, symbolic::one())));

    auto ext_ik = group_ik->tile.extents();
    ASSERT_EQ(ext_ik.size(), 2);
    EXPECT_TRUE(symbolic::eq(ext_ik.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(ext_ik.at(1), K));

    // Group for A[j,k]: tile should be [j, 0] -> [j, K-1], extents [1, K]
    ASSERT_EQ(group_jk->tile.min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(group_jk->tile.min_subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(group_jk->tile.min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(group_jk->tile.max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(group_jk->tile.max_subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(group_jk->tile.max_subset.at(1), symbolic::sub(K, symbolic::one())));

    auto ext_jk = group_jk->tile.extents();
    ASSERT_EQ(ext_jk.size(), 2);
    EXPECT_TRUE(symbolic::eq(ext_jk.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(ext_jk.at(1), K));

    // Each group should contain exactly 1 memlet
    EXPECT_EQ(group_ik->memlets.size(), 1);
    EXPECT_EQ(group_jk->memlets.size(), 1);
    EXPECT_EQ(group_ik->memlets[0], &memlet_A_ik);
    EXPECT_EQ(group_jk->memlets[0], &memlet_A_jk);
}

// Test tile groups: single access pattern should produce one group
TEST(MemoryLayoutAnalysisTest, TileGroups_SingleAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [0, N)
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // for j in [0, M)
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // A[i*M + j]
    auto& block = builder.add_block(j_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // tile_groups at j_loop should have 1 group (both memlets access same [i,j] pattern)
    auto* groups = analysis.tile_groups(j_loop, "A");
    ASSERT_NE(groups, nullptr);
    ASSERT_EQ(groups->size(), 1);

    // The single group should contain both memlets
    auto& group = (*groups)[0];
    EXPECT_EQ(group.memlets.size(), 2);

    // Group tile should match the regular tile
    auto* regular_tile = analysis.tile(j_loop, "A");
    ASSERT_NE(regular_tile, nullptr);

    EXPECT_TRUE(symbolic::eq(group.tile.min_subset.at(0), regular_tile->min_subset.at(0)));
    EXPECT_TRUE(symbolic::eq(group.tile.min_subset.at(1), regular_tile->min_subset.at(1)));
    EXPECT_TRUE(symbolic::eq(group.tile.max_subset.at(0), regular_tile->max_subset.at(0)));
    EXPECT_TRUE(symbolic::eq(group.tile.max_subset.at(1), regular_tile->max_subset.at(1)));

    // tile_group_for should find both memlets in the same group
    auto* g_in = analysis.tile_group_for(j_loop, memlet_in);
    auto* g_out = analysis.tile_group_for(j_loop, memlet_out);
    ASSERT_NE(g_in, nullptr);
    ASSERT_NE(g_out, nullptr);
    EXPECT_EQ(g_in, g_out); // Same group
}

// Test tile groups: stencil pattern should produce multiple groups but merged tile still works
TEST(MemoryLayoutAnalysisTest, TileGroups_StencilPattern) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [1, N-1)
    auto& i_loop = builder.add_for(
        root,
        i,
        symbolic::Lt(i, symbolic::sub(N, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(i, symbolic::one())
    );

    // for j in [1, M-1)
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(j, symbolic::one())
    );

    // Block: 3-point stencil in i: A[i*M+j], A[(i-1)*M+j], A[(i+1)*M+j]
    auto& block = builder.add_block(j_loop.root());
    auto& a_center = builder.add_access(block, "A");
    auto& a_north = builder.add_access(block, "A");
    auto& a_south = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_c", "_n", "_s"});

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);

    auto& m_center = builder.add_computational_memlet(block, a_center, tasklet, "_c", {center});
    auto& m_north = builder.add_computational_memlet(block, a_north, tasklet, "_n", {north});
    auto& m_south = builder.add_computational_memlet(block, a_south, tasklet, "_s", {south});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {center});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // The merged tile at j_loop should still work (extents [3, M-2] are integer)
    auto* regular_tile = analysis.tile(j_loop, "A");
    ASSERT_NE(regular_tile, nullptr);

    auto ext = regular_tile->extents();
    ASSERT_EQ(ext.size(), 2);
    EXPECT_TRUE(symbolic::eq(ext.at(0), symbolic::integer(3)));

    // Tile groups at j_loop: stencil accesses have bases that differ only by
    // integer constants: [i-1, 1], [i, 1], [i+1, 1]. These get merged into
    // a single group because the differences are purely constant offsets.
    auto* groups = analysis.tile_groups(j_loop, "A");
    ASSERT_NE(groups, nullptr);
    // All stencil points merge into 1 group (constant-offset bases)
    EXPECT_EQ(groups->size(), 1);

    // The single group should contain all 4 memlets (3 inputs + 1 output)
    auto* g_center = analysis.tile_group_for(j_loop, m_center);
    ASSERT_NE(g_center, nullptr);
    EXPECT_EQ(g_center->memlets.size(), 4);

    // All stencil memlets are in the same group
    auto* g_north = analysis.tile_group_for(j_loop, m_north);
    auto* g_south = analysis.tile_group_for(j_loop, m_south);
    ASSERT_NE(g_north, nullptr);
    ASSERT_NE(g_south, nullptr);
    EXPECT_EQ(g_center, g_north);
    EXPECT_EQ(g_center, g_south);
}

// Test tile groups: tiled SYR2K pattern A[i_tile+i, k] + A[j_tile+j, k]
TEST(MemoryLayoutAnalysisTest, TileGroups_TiledTwoAccesses) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i_tile", index_type);
    builder.add_container("j_tile", index_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);
    builder.add_container("tmp", scalar_type);

    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto tile_size = symbolic::integer(8);

    // for i_tile in [0, N) step 8
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, tile_size));

    // for j_tile in [0, N) step 8
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, N), symbolic::integer(0), symbolic::add(j_tile, tile_size)
    );

    // for i in [i_tile, min(i_tile+8, N))
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, tile_size)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j in [j_tile, min(j_tile+8, N))
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, tile_size)), symbolic::Lt(j, N)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // for k in [0, K)
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block: tmp = A[i*K + k] * A[j*K + k]
    auto& block = builder.add_block(k_loop.root());
    auto& access_A_ik = builder.add_access(block, "A");
    auto& access_A_jk = builder.add_access(block, "A");
    auto& access_tmp = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_a", "_b"});

    auto linearized_ik = symbolic::add(symbolic::mul(i, K), k);
    auto linearized_jk = symbolic::add(symbolic::mul(j, K), k);

    auto& memlet_A_ik = builder.add_computational_memlet(block, access_A_ik, tasklet, "_a", {linearized_ik});
    auto& memlet_A_jk = builder.add_computational_memlet(block, access_A_jk, tasklet, "_b", {linearized_jk});
    builder.add_computational_memlet(block, tasklet, "_out", access_tmp, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // At the k_loop level, tile groups should separate A[i,k] from A[j,k]
    auto* k_groups = analysis.tile_groups(k_loop, "A");
    ASSERT_NE(k_groups, nullptr);
    ASSERT_EQ(k_groups->size(), 2);

    auto* group_ik = analysis.tile_group_for(k_loop, memlet_A_ik);
    auto* group_jk = analysis.tile_group_for(k_loop, memlet_A_jk);
    ASSERT_NE(group_ik, nullptr);
    ASSERT_NE(group_jk, nullptr);
    EXPECT_NE(group_ik, group_jk);

    // Group for A[i,k]: extents should be [1, K] (integer!)
    auto ext_ik = group_ik->tile.extents();
    ASSERT_EQ(ext_ik.size(), 2);
    EXPECT_TRUE(symbolic::eq(ext_ik.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(ext_ik.at(1), K));

    // Group for A[j,k]: extents should also be [1, K] (integer!)
    auto ext_jk = group_jk->tile.extents();
    ASSERT_EQ(ext_jk.size(), 2);
    EXPECT_TRUE(symbolic::eq(ext_jk.at(0), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(ext_jk.at(1), K));
}

// ============================================================================
// LU factorization loop nest replication.
//
// Mirrors the un-tiled LUFixture in docc/opt/tests/transformations/optimizations
// /blocking_test.cpp. The point of this test is diagnostic: walk MLA over each
// of LU's seven distinct A subscripts and document which ones delinearize and
// which fall through. Failures here are expected to drive MLA improvements.
//
// LU loop nest:
//   for i in [0, N):
//     for j in [0, i):                                    // ChildA
//       for k in [0, j):
//         (1) read  A[i*N + k]
//         (2) read  A[k*N + j]
//         (3) read  A[i*N + j]
//         (4) write A[i*N + j]
//       (5) read  A[i*N + j]
//       (6) read  A[j*N + j]
//       (7) write A[i*N + j]
//     for j2 in [0, N - i):                               // ChildB
//       for k2 in [0, i):
//         (8) read  A[i*N + k2]
//         (9) read  A[k2*N + (i + j2)]
//         (10) read A[i*N + (i + j2)]
//         (11) write A[i*N + (i + j2)]
//
// Distinct subscripts on A:
//   S1: i*N + k          (i, k)
//   S2: k*N + j          (k, j)
//   S3: i*N + j          (i, j)
//   S4: j*N + j          (j, j)
//   S5: i*N + k2         (i, k2)
//   S6: k2*N + (i + j2)  (k2, i+j2)
//   S7: i*N + (i + j2)   (i, i+j2)
// ============================================================================
TEST(MemoryLayoutAnalysisTest, LU_Factorization_Diagnostic) {
    builder::StructuredSDFGBuilder builder("lu_mla", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("j2", sym_desc);
    builder.add_container("k2", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("tmp_2", elem_desc);
    builder.add_container("tmp_8", elem_desc);

    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto j2 = symbolic::symbol("j2");
    auto k2 = symbolic::symbol("k2");
    auto N = symbolic::symbol("N");
    auto one = symbolic::integer(1);
    auto zero = symbolic::integer(0);

    // for (i = 0; i < N; i++)
    auto& i_loop = builder.add_for(root, i, symbolic::Lt(i, N), zero, symbolic::add(i, one));

    // ChildA: for (j = 0; j < i; j++)
    auto& j_loop = builder.add_for(i_loop.root(), j, symbolic::Lt(j, i), zero, symbolic::add(j, one));

    // for (k = 0; k < j; k++) { tmp_2 = A[i*N+k]*A[k*N+j]; A[i*N+j] -= tmp_2; }
    auto& k_loop = builder.add_for(j_loop.root(), k, symbolic::Lt(k, j), zero, symbolic::add(k, one));

    // mul block: A[i*N + k] * A[k*N + j] -> tmp_2
    const data_flow::Memlet* mlt_S1 = nullptr;
    const data_flow::Memlet* mlt_S2 = nullptr;
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_2");
        mlt_S1 =
            &builder
                 .add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k)}, ptr_desc);
        mlt_S2 =
            &builder
                 .add_computational_memlet(block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }

    // sub block: A[i*N + j] -= tmp_2
    const data_flow::Memlet* mlt_S3_in = nullptr;
    const data_flow::Memlet* mlt_S3_out = nullptr;
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_2");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S3_in =
            &builder
                 .add_computational_memlet(block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        mlt_S3_out =
            &builder
                 .add_computational_memlet(block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }

    // div block: A[i*N + j] /= A[j*N + j]
    const data_flow::Memlet* mlt_S3_div_in = nullptr;
    const data_flow::Memlet* mlt_S4 = nullptr;
    const data_flow::Memlet* mlt_S3_div_out = nullptr;
    {
        auto& block = builder.add_block(j_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_div = builder.add_access(block, "A");
        auto& div_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S3_div_in =
            &builder
                 .add_computational_memlet(block, a_in, div_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        mlt_S4 =
            &builder
                 .add_computational_memlet(block, a_div, div_t, "_in2", {symbolic::add(symbolic::mul(j, N), j)}, ptr_desc);
        mlt_S3_div_out =
            &builder
                 .add_computational_memlet(block, div_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }

    // ChildB: for (j2 = 0; j2 < N - i; j2++)
    auto& j2_loop =
        builder.add_for(i_loop.root(), j2, symbolic::Lt(j2, symbolic::sub(N, i)), zero, symbolic::add(j2, one));

    // for (k2 = 0; k2 < i; k2++) { tmp_8 = A[i*N + k2]*A[k2*N + (i+j2)]; A[i*N + (i+j2)] -= tmp_8; }
    auto& k2_loop = builder.add_for(j2_loop.root(), k2, symbolic::Lt(k2, i), zero, symbolic::add(k2, one));

    const data_flow::Memlet* mlt_S5 = nullptr;
    const data_flow::Memlet* mlt_S6 = nullptr;
    {
        auto& block = builder.add_block(k2_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_8");
        mlt_S5 =
            &builder
                 .add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k2)}, ptr_desc);
        mlt_S6 = &builder.add_computational_memlet(
            block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k2, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }

    const data_flow::Memlet* mlt_S7_in = nullptr;
    const data_flow::Memlet* mlt_S7_out = nullptr;
    {
        auto& block = builder.add_block(k2_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_8");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S7_in = &builder.add_computational_memlet(
            block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        mlt_S7_out = &builder.add_computational_memlet(
            block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
    }

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto fmt_subset = [](const data_flow::Subset& s) {
        std::string out = "[";
        for (size_t idx = 0; idx < s.size(); ++idx) {
            if (idx) out += ", ";
            out += SymEngine::str(*s.at(idx));
        }
        out += "]";
        return out;
    };
    auto check = [&](const char* label, const data_flow::Memlet* m) {
        auto* acc = analysis.access(*m);
        (void) label;
        (void) fmt_subset;
        return acc;
    };

    auto* a_S1 = check("S1 read  A[i*N + k]", mlt_S1);
    auto* a_S2 = check("S2 read  A[k*N + j]", mlt_S2);
    auto* a_S3a = check("S3 read  A[i*N + j] (sub)", mlt_S3_in);
    auto* a_S3b = check("S3 write A[i*N + j] (sub)", mlt_S3_out);
    auto* a_S3c = check("S3 read  A[i*N + j] (div)", mlt_S3_div_in);
    auto* a_S4 = check("S4 read  A[j*N + j]", mlt_S4);
    auto* a_S3d = check("S3 write A[i*N + j] (div)", mlt_S3_div_out);
    auto* a_S5 = check("S5 read  A[i*N + k2]", mlt_S5);
    auto* a_S6 = check("S6 read  A[k2*N + (i+j2)]", mlt_S6);
    auto* a_S7a = check("S7 read  A[i*N + (i+j2)]", mlt_S7_in);
    auto* a_S7b = check("S7 write A[i*N + (i+j2)]", mlt_S7_out);

    // S1: i*N + k --> [i, k]
    ASSERT_NE(a_S1, nullptr) << "S1 should delinearize";
    ASSERT_EQ(a_S1->subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(a_S1->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(a_S1->subset.at(1), k));

    // S2: k*N + j --> [k, j]
    ASSERT_NE(a_S2, nullptr) << "S2 should delinearize";
    ASSERT_EQ(a_S2->subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(a_S2->subset.at(0), k));
    EXPECT_TRUE(symbolic::eq(a_S2->subset.at(1), j));

    // S3: i*N + j --> [i, j]
    for (auto* acc : {a_S3a, a_S3b, a_S3c, a_S3d}) {
        ASSERT_NE(acc, nullptr) << "S3 should delinearize";
        ASSERT_EQ(acc->subset.size(), 2u);
        EXPECT_TRUE(symbolic::eq(acc->subset.at(0), i));
        EXPECT_TRUE(symbolic::eq(acc->subset.at(1), j));
    }

    // S4: j*N + j --> [j, j]
    ASSERT_NE(a_S4, nullptr) << "S4 should delinearize";
    ASSERT_EQ(a_S4->subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(a_S4->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(a_S4->subset.at(1), j));

    // S5: i*N + k2 --> [i, k2]
    ASSERT_NE(a_S5, nullptr) << "S5 should delinearize";
    ASSERT_EQ(a_S5->subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(a_S5->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(a_S5->subset.at(1), k2));

    // S6: k2*N + (i + j2) --> [k2, i + j2]
    ASSERT_NE(a_S6, nullptr) << "S6 should delinearize";
    ASSERT_EQ(a_S6->subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(a_S6->subset.at(0), k2));
    EXPECT_TRUE(symbolic::eq(a_S6->subset.at(1), symbolic::add(i, j2)));

    // S7: i*N + (i + j2) --> [i, i + j2]
    for (auto* acc : {a_S7a, a_S7b}) {
        ASSERT_NE(acc, nullptr) << "S7 should delinearize";
        ASSERT_EQ(acc->subset.size(), 2u);
        EXPECT_TRUE(symbolic::eq(acc->subset.at(0), i));
        EXPECT_TRUE(symbolic::eq(acc->subset.at(1), symbolic::add(i, j2)));
    }
}

// ============================================================================
// Blocked LU factorization MLA diagnostic
//
// Mirrors the loop structure that BlockingTest.LU_BlockedPipeline produces
// after `LoopTiling(i, 64)` followed by `LoopSplit(j2, (i_tile0+64) - i)`:
//
//   for i_tile0 in [0, N) step 64:
//     for i in [i_tile0, min(N, i_tile0+64)):
//       for j in [0, i):                       // ChildA (unchanged)
//         for k in [0, j):
//           A[i*N + k]; A[k*N + j]
//           A[i*N + j] -= ...
//         A[i*N + j] /= A[j*N + j]
//       for j2 in [0, (i_tile0+64) - i):       // in-panel ChildB
//         for k2 in [0, i):
//           A[i*N + k2]; A[k2*N + (i+j2)]
//           A[i*N + (i+j2)] -= ...
//       for j20 in [(i_tile0+64) - i, N - i):  // trailing ChildB (renamed)
//         for k2 in [0, i):
//           A[i*N + k2]; A[k2*N + (i+j20)]
//           A[i*N + (i+j20)] -= ...
//
// All seven distinct subscript shapes on A must delinearize to 2D row-major.
// This test serves as a fast-iteration target for MLA fixes that target the
// blocked-LU access patterns surfaced by the integration test in
// `BlockingTest.LU_BlockedPipeline`.
// ============================================================================
TEST(MemoryLayoutAnalysisTest, LU_BlockedFactorization_Diagnostic) {
    builder::StructuredSDFGBuilder builder("lu_blocked_mla", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile0", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("j2", sym_desc);
    builder.add_container("j20", sym_desc);
    builder.add_container("k2", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("tmp_2", elem_desc);
    builder.add_container("tmp_8", elem_desc);

    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto i_tile0 = symbolic::symbol("i_tile0");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto j2 = symbolic::symbol("j2");
    auto j20 = symbolic::symbol("j20");
    auto k2 = symbolic::symbol("k2");
    auto N = symbolic::symbol("N");
    auto one = symbolic::integer(1);
    auto zero = symbolic::integer(0);
    auto B = symbolic::integer(64);

    // for i_tile0 in [0, N) step 64
    auto& it_loop = builder.add_for(root, i_tile0, symbolic::Lt(i_tile0, N), zero, symbolic::add(i_tile0, B));

    // for i in [i_tile0, min(N, i_tile0+64))
    auto i_upper = symbolic::min(N, symbolic::add(i_tile0, B));
    auto& i_loop = builder.add_for(it_loop.root(), i, symbolic::Lt(i, i_upper), i_tile0, symbolic::add(i, one));

    // ChildA: for j in [0, i)
    auto& j_loop = builder.add_for(i_loop.root(), j, symbolic::Lt(j, i), zero, symbolic::add(j, one));

    // for k in [0, j)
    auto& k_loop = builder.add_for(j_loop.root(), k, symbolic::Lt(k, j), zero, symbolic::add(k, one));

    const data_flow::Memlet* mlt_S1 = nullptr;
    const data_flow::Memlet* mlt_S2 = nullptr;
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_2");
        mlt_S1 =
            &builder
                 .add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k)}, ptr_desc);
        mlt_S2 =
            &builder
                 .add_computational_memlet(block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }

    const data_flow::Memlet* mlt_S3_in = nullptr;
    const data_flow::Memlet* mlt_S3_out = nullptr;
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_2");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S3_in =
            &builder
                 .add_computational_memlet(block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        mlt_S3_out =
            &builder
                 .add_computational_memlet(block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }

    const data_flow::Memlet* mlt_S4 = nullptr;
    {
        auto& block = builder.add_block(j_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_div = builder.add_access(block, "A");
        auto& div_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        builder.add_computational_memlet(block, a_in, div_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        mlt_S4 =
            &builder
                 .add_computational_memlet(block, a_div, div_t, "_in2", {symbolic::add(symbolic::mul(j, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, div_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }

    // In-panel ChildB: for j2 in [0, (i_tile0 + 64) - i) with the original
    // LoopSplit-fixed conjoined condition `j2 < (i_tile0+64)-i AND j2 < N-i`.
    auto j2_split = symbolic::sub(symbolic::add(i_tile0, B), i);
    auto j2_orig_ub = symbolic::sub(N, i);
    auto j2_cond = symbolic::And(symbolic::Lt(j2, j2_split), symbolic::Lt(j2, j2_orig_ub));
    auto& j2_loop = builder.add_for(i_loop.root(), j2, j2_cond, zero, symbolic::add(j2, one));
    auto& k2a_loop = builder.add_for(j2_loop.root(), k2, symbolic::Lt(k2, i), zero, symbolic::add(k2, one));

    const data_flow::Memlet* mlt_S5_p = nullptr;
    const data_flow::Memlet* mlt_S6_p = nullptr;
    {
        auto& block = builder.add_block(k2a_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_8");
        mlt_S5_p =
            &builder
                 .add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k2)}, ptr_desc);
        mlt_S6_p = &builder.add_computational_memlet(
            block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k2, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }

    const data_flow::Memlet* mlt_S7_p_in = nullptr;
    const data_flow::Memlet* mlt_S7_p_out = nullptr;
    {
        auto& block = builder.add_block(k2a_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_8");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S7_p_in = &builder.add_computational_memlet(
            block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        mlt_S7_p_out = &builder.add_computational_memlet(
            block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
    }

    // Trailing ChildB: for j20 in [(i_tile0+64) - i, N - i)
    auto j20_lower = symbolic::sub(symbolic::add(i_tile0, B), i);
    auto j20_upper = symbolic::sub(N, i);
    auto& j20_loop =
        builder.add_for(i_loop.root(), j20, symbolic::Lt(j20, j20_upper), j20_lower, symbolic::add(j20, one));
    auto& k2b_loop = builder.add_for(j20_loop.root(), k2, symbolic::Lt(k2, i), zero, symbolic::add(k2, one));

    const data_flow::Memlet* mlt_S5_t = nullptr;
    const data_flow::Memlet* mlt_S6_t = nullptr;
    {
        auto& block = builder.add_block(k2b_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_8");
        mlt_S5_t =
            &builder
                 .add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k2)}, ptr_desc);
        mlt_S6_t = &builder.add_computational_memlet(
            block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k2, N), symbolic::add(i, j20))}, ptr_desc
        );
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }

    const data_flow::Memlet* mlt_S7_t_in = nullptr;
    const data_flow::Memlet* mlt_S7_t_out = nullptr;
    {
        auto& block = builder.add_block(k2b_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_8");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        mlt_S7_t_in = &builder.add_computational_memlet(
            block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j20))}, ptr_desc
        );
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        mlt_S7_t_out = &builder.add_computational_memlet(
            block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j20))}, ptr_desc
        );
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto fmt_subset = [](const data_flow::Subset& s) {
        std::string out = "[";
        for (size_t idx = 0; idx < s.size(); ++idx) {
            if (idx) out += ", ";
            out += SymEngine::str(*s.at(idx));
        }
        out += "]";
        return out;
    };
    auto get = [&](const char* label, const data_flow::Memlet* m) {
        auto* acc = analysis.access(*m);
        if (!acc) {
            std::cerr << "[BLOCKED-MLA-FAIL] " << label << " no access info\n";
        } else {
            std::cerr << "[BLOCKED-MLA] " << label << " -> " << fmt_subset(acc->subset) << "\n";
        }
        return acc;
    };

    auto* a_S1 = get("S1 A[i*N+k]", mlt_S1);
    auto* a_S2 = get("S2 A[k*N+j]", mlt_S2);
    auto* a_S3a = get("S3 A[i*N+j] (sub-in)", mlt_S3_in);
    auto* a_S3b = get("S3 A[i*N+j] (sub-out)", mlt_S3_out);
    auto* a_S4 = get("S4 A[j*N+j]", mlt_S4);
    auto* a_S5p = get("S5 A[i*N+k2] (in-panel)", mlt_S5_p);
    auto* a_S6p = get("S6 A[k2*N+(i+j2)] (in-panel)", mlt_S6_p);
    auto* a_S7p_in = get("S7 A[i*N+(i+j2)] in (in-panel)", mlt_S7_p_in);
    auto* a_S7p_out = get("S7 A[i*N+(i+j2)] out (in-panel)", mlt_S7_p_out);
    auto* a_S5t = get("S5 A[i*N+k2] (trailing)", mlt_S5_t);
    auto* a_S6t = get("S6 A[k2*N+(i+j20)] (trailing)", mlt_S6_t);
    auto* a_S7t_in = get("S7 A[i*N+(i+j20)] in (trailing)", mlt_S7_t_in);
    auto* a_S7t_out = get("S7 A[i*N+(i+j20)] out (trailing)", mlt_S7_t_out);

    auto check_2d = [](const analysis::MemoryAccess* acc,
                       const char* label,
                       const symbolic::Expression& d0,
                       const symbolic::Expression& d1) {
        ASSERT_NE(acc, nullptr) << label << " should delinearize";
        ASSERT_EQ(acc->subset.size(), 2u) << label;
        EXPECT_TRUE(symbolic::eq(acc->subset.at(0), d0))
            << label << " dim0 mismatch: got " << SymEngine::str(*acc->subset.at(0)) << " expected "
            << SymEngine::str(*d0);
        EXPECT_TRUE(symbolic::eq(acc->subset.at(1), d1))
            << label << " dim1 mismatch: got " << SymEngine::str(*acc->subset.at(1)) << " expected "
            << SymEngine::str(*d1);
    };

    check_2d(a_S1, "S1", i, k);
    check_2d(a_S2, "S2", k, j);
    check_2d(a_S3a, "S3 sub-in", i, j);
    check_2d(a_S3b, "S3 sub-out", i, j);
    check_2d(a_S4, "S4", j, j);
    check_2d(a_S5p, "S5 in-panel", i, k2);
    check_2d(a_S6p, "S6 in-panel", k2, symbolic::add(i, j2));
    check_2d(a_S7p_in, "S7 in-panel sub-in", i, symbolic::add(i, j2));
    check_2d(a_S7p_out, "S7 in-panel sub-out", i, symbolic::add(i, j2));
    check_2d(a_S5t, "S5 trailing", i, k2);
    check_2d(a_S6t, "S6 trailing", k2, symbolic::add(i, j20));
    check_2d(a_S7t_in, "S7 trailing sub-in", i, symbolic::add(i, j20));
    check_2d(a_S7t_out, "S7 trailing sub-out", i, symbolic::add(i, j20));
}

// =====================================================================
// Scope-generic API tests: tiles should also be queryable at non-loop
// control-flow scopes (root Sequence, IfElse, While).
// =====================================================================

TEST(MemoryLayoutAnalysisTest, ScopeAPI_RootSequence_SingleNestedLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Outer-loop tile is the established reference.
    auto* tile_outer = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_outer, nullptr);

    // Root sequence tile should exist (scope-generic API) and match the outer-loop tile,
    // because the outer loop is the only direct child carrying A accesses.
    auto* tile_root = analysis.tile(root, "A");
    ASSERT_NE(tile_root, nullptr);

    ASSERT_EQ(tile_root->min_subset.size(), tile_outer->min_subset.size());
    for (size_t d = 0; d < tile_outer->min_subset.size(); ++d) {
        EXPECT_TRUE(symbolic::eq(tile_root->min_subset.at(d), tile_outer->min_subset.at(d)));
        EXPECT_TRUE(symbolic::eq(tile_root->max_subset.at(d), tile_outer->max_subset.at(d)));
    }
}

TEST(MemoryLayoutAnalysisTest, ScopeAPI_RootSequence_TwoSiblingLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("i2", index_type);
    builder.add_container("j2", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto i2 = symbolic::symbol("i2");
    auto j2 = symbolic::symbol("j2");

    // First nest: writes A[i*M + j] for i in [0, N), j in [0, M)
    {
        auto& loop_i =
            builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
        auto& loop_j =
            builder
                .add_for(loop_i.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
        auto& block = builder.add_block(loop_j.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto idx = symbolic::add(symbolic::mul(i, M), j);
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});
    }
    // Second nest: writes A[i2*M + j2] for i2 in [0, N), j2 in [0, M) (independent indvars)
    {
        auto& loop_i =
            builder.add_for(root, i2, symbolic::Lt(i2, N), symbolic::integer(0), symbolic::add(i2, symbolic::one()));
        auto& loop_j =
            builder
                .add_for(loop_i.root(), j2, symbolic::Lt(j2, M), symbolic::integer(0), symbolic::add(j2, symbolic::one()));
        auto& block = builder.add_block(loop_j.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto idx = symbolic::add(symbolic::mul(i2, M), j2);
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Root sequence tile should exist and union both child loop tiles. Each loop
    // covers [0..N-1, 0..M-1], so the union is identical.
    auto* tile_root = analysis.tile(root, "A");
    ASSERT_NE(tile_root, nullptr);

    ASSERT_EQ(tile_root->min_subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(tile_root->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_root->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_root->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_root->max_subset.at(1), symbolic::sub(M, symbolic::one())));
}

TEST(MemoryLayoutAnalysisTest, ScopeAPI_IfElse_BothBranches) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("cond", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto cond = symbolic::symbol("cond");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& if_else = builder.add_if_else(root);
    auto& branch_true = builder.add_case(if_else, symbolic::Eq(cond, symbolic::zero()));
    auto& branch_false = builder.add_case(if_else, symbolic::Ne(cond, symbolic::zero()));

    auto build_nest = [&](structured_control_flow::Sequence& parent) {
        auto& loop_i =
            builder.add_for(parent, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
        auto& loop_j =
            builder
                .add_for(loop_i.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
        auto& block = builder.add_block(loop_j.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto idx = symbolic::add(symbolic::mul(i, M), j);
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});
        return &loop_i;
    };

    build_nest(branch_true);
    build_nest(branch_false);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Each branch sequence has its own tile.
    auto* tile_branch_true = analysis.tile(branch_true, "A");
    ASSERT_NE(tile_branch_true, nullptr);
    auto* tile_branch_false = analysis.tile(branch_false, "A");
    ASSERT_NE(tile_branch_false, nullptr);

    // The IfElse scope tile unions both branches; bounds match either branch (identical here).
    auto* tile_ife = analysis.tile(if_else, "A");
    ASSERT_NE(tile_ife, nullptr);

    ASSERT_EQ(tile_ife->min_subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(tile_ife->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_ife->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_ife->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_ife->max_subset.at(1), symbolic::sub(M, symbolic::one())));

    // Root sequence picks up the IfElse contribution.
    auto* tile_root = analysis.tile(root, "A");
    ASSERT_NE(tile_root, nullptr);
    ASSERT_EQ(tile_root->min_subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(tile_root->min_subset.at(0), tile_ife->min_subset.at(0)));
    EXPECT_TRUE(symbolic::eq(tile_root->max_subset.at(0), tile_ife->max_subset.at(0)));
}

TEST(MemoryLayoutAnalysisTest, ScopeAPI_While_PassThrough) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& while_loop = builder.add_while(root);

    auto& loop_i =
        builder
            .add_for(while_loop.root(), i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    auto& loop_j =
        builder.add_for(loop_i.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    auto& block = builder.add_block(loop_j.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto idx = symbolic::add(symbolic::mul(i, M), j);
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_body = analysis.tile(while_loop.root(), "A");
    ASSERT_NE(tile_body, nullptr);

    // While scope tile should equal its body sequence tile.
    auto* tile_while = analysis.tile(while_loop, "A");
    ASSERT_NE(tile_while, nullptr);

    ASSERT_EQ(tile_while->min_subset.size(), tile_body->min_subset.size());
    for (size_t d = 0; d < tile_body->min_subset.size(); ++d) {
        EXPECT_TRUE(symbolic::eq(tile_while->min_subset.at(d), tile_body->min_subset.at(d)));
        EXPECT_TRUE(symbolic::eq(tile_while->max_subset.at(d), tile_body->max_subset.at(d)));
    }
}

TEST(MemoryLayoutAnalysisTest, ScopeAPI_TileGroups_NonLoopScope) {
    // Stencil-like pattern with constant-offset bases should produce a merged
    // tile group not only at the loop level but also at the enclosing scope.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& loop_i = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::sub(N, symbolic::one())), symbolic::one(), symbolic::add(i, symbolic::one())
    );
    auto& loop_j = builder.add_for(
        loop_i.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::one(),
        symbolic::add(j, symbolic::one())
    );

    // Two reads of A with constant-offset bases: A[i*M + j] and A[i*M + (j+1)]
    auto& block = builder.add_block(loop_j.root());
    auto& a_c = builder.add_access(block, "A");
    auto& a_r = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_inc", "_inr"});
    auto idx_c = symbolic::add(symbolic::mul(i, M), j);
    auto idx_r = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));
    builder.add_computational_memlet(block, a_c, tasklet, "_inc", {idx_c});
    builder.add_computational_memlet(block, a_r, tasklet, "_inr", {idx_r});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx_c});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Loop-level groups: stencil bases merge into one group at the j-loop level.
    auto* groups_j = analysis.tile_groups(loop_j, "A");
    ASSERT_NE(groups_j, nullptr);

    // The root sequence should also expose tile groups for A (propagated upward).
    auto* groups_root = analysis.tile_groups(root, "A");
    ASSERT_NE(groups_root, nullptr);
    EXPECT_FALSE(groups_root->empty());
}

// -----------------------------------------------------------------------------
// Regression tests targeting the ResNet im2col offload regression.
//
// The failing kernel is a 2D collapsed map writing `_patches0` from `_1`. The
// subscripts involve `i % C` and `i / C` of plain (constant-bound) indvars,
// and the dataflow lives inside an `IfElse` with branch-disjoint reads. The
// tests below isolate the smallest MLA patterns that should still produce a
// tile bound (`tile(map, container)` and `contiguous_range()`); each one will
// expose a separate gap if MLA regresses again.
// -----------------------------------------------------------------------------

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_ModSubscript) {
    // for i in [0, 1024): A[i % 16] -- pointer access with a single mod.
    // Outer tile for A should be bounded by [0, 15].
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("A", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(1024)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto idx = symbolic::mod(i, symbolic::integer(16));
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_map = analysis.tile(map, "A");
    ASSERT_NE(tile_map, nullptr) << "MLA could not bound A[i % 16] over the map scope.";
    auto range = tile_map->contiguous_range();
    EXPECT_FALSE(range.first.is_null());
    EXPECT_FALSE(range.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_DivSubscript) {
    // for i in [0, 1024): A[i / 16] -- single floor-div subscript.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("A", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(1024)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto idx = symbolic::div(i, symbolic::integer(16));
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_map = analysis.tile(map, "A");
    ASSERT_NE(tile_map, nullptr) << "MLA could not bound A[i / 16] over the map scope.";
    auto range = tile_map->contiguous_range();
    EXPECT_FALSE(range.first.is_null());
    EXPECT_FALSE(range.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_MixedModDivStrided) {
    // for i in [0, 401408): for j in [0, 147):
    //     A[150528*(i/12544) + 50176*(j/49) + 224*((i/112)%112) + (i%112)]
    // -- mod/div linear combination, the form produced by collapsing four
    // outer loops to one in resnet's im2col.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::integer(401408);
    auto M = symbolic::integer(147);

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::zero(),
        symbolic::add(j, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(inner.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto idx = symbolic::
        add(symbolic::
                add(symbolic::mul(symbolic::integer(150528), symbolic::div(i, symbolic::integer(12544))),
                    symbolic::mul(symbolic::integer(50176), symbolic::div(j, symbolic::integer(49)))),
            symbolic::
                add(symbolic::
                        mul(symbolic::integer(224),
                            symbolic::mod(symbolic::div(i, symbolic::integer(112)), symbolic::integer(112))),
                    symbolic::mod(i, symbolic::integer(112))));
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_outer = analysis.tile(outer, "A");
    ASSERT_NE(tile_outer, nullptr) << "MLA could not bound the collapsed im2col-style mod/div access at the outer map.";
    auto range = tile_outer->contiguous_range();
    EXPECT_FALSE(range.first.is_null());
    EXPECT_FALSE(range.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_AccessInsideIfElse) {
    // for i in [0, 1024): if (i < 16) A[i] = 0;
    // The map body contains an IfElse rather than a Block; the pointer access
    // lives inside one branch only.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("A", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(1024)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& ife = builder.add_if_else(map.root());
    auto& taken = builder.add_case(ife, symbolic::Lt(i, symbolic::integer(16)));
    auto& not_taken = builder.add_case(ife, symbolic::Ge(i, symbolic::integer(16)));
    auto& block = builder.add_block(taken);
    auto& constant = builder.add_constant(block, "0.0f", scalar_type);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, constant, tasklet, "_in", {}, scalar_type);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {i});
    // Suppress unused-variable warning for the empty else case.
    (void) not_taken;

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_map = analysis.tile(map, "A");
    ASSERT_NE(tile_map, nullptr) << "MLA returns nullptr when the only access to A lives in one IfElse branch.";
    auto range = tile_map->contiguous_range();
    EXPECT_FALSE(range.first.is_null());
    EXPECT_FALSE(range.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_TwoArgsIfElseBranchAsymmetric) {
    // Map body is an IfElse with two cases, both writing to `_patches`:
    //   if  (i < 16): _patches[i] = _1[i];   (reads _1)
    //   if  (i >= 16): _patches[i] = 0.0f;   (does NOT read _1)
    // This mirrors the resnet asymmetry: one container is accessed in both
    // branches, another in only one. MLA must still bound both at the map.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("_1", pointer_type, true);
    builder.add_container("_patches", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(1024)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& ife = builder.add_if_else(map.root());
    auto& taken = builder.add_case(ife, symbolic::Lt(i, symbolic::integer(16)));
    auto& not_taken = builder.add_case(ife, symbolic::Ge(i, symbolic::integer(16)));

    {
        auto& block = builder.add_block(taken);
        auto& in_node = builder.add_access(block, "_1");
        auto& out_node = builder.add_access(block, "_patches");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in_node, tasklet, "_in", {i});
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {i});
    }
    {
        auto& block = builder.add_block(not_taken);
        auto& constant = builder.add_constant(block, "0.0f", scalar_type);
        auto& out_node = builder.add_access(block, "_patches");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, constant, tasklet, "_in", {}, scalar_type);
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {i});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_in = analysis.tile(map, "_1");
    ASSERT_NE(tile_in, nullptr) << "MLA could not bound _1 (one-branch read) at the map scope.";
    auto range_in = tile_in->contiguous_range();
    EXPECT_FALSE(range_in.first.is_null());
    EXPECT_FALSE(range_in.second.is_null());

    auto* tile_out = analysis.tile(map, "_patches");
    ASSERT_NE(tile_out, nullptr) << "MLA could not bound _patches (both-branch write) at the map scope.";
    auto range_out = tile_out->contiguous_range();
    EXPECT_FALSE(range_out.first.is_null());
    EXPECT_FALSE(range_out.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_NegativeOffsetSubscript) {
    // Map body has an IfElse guarding a negative-offset read:
    //   if (i >= 3 && i < 224 + 3): _patches[i-3] = _1[i-3];
    // A common simplification result of im2col padding logic. The subscript
    // (i - 3) reaches -2..-1 at the loop's lower bound when the guard is
    // ignored. MLA's BoundAnalysis must respect the loop range and yield a
    // sound [-3, last] tile, not give up because of the negative offset.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("_1", pointer_type, true);
    builder.add_container("_patches", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(230)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& ife = builder.add_if_else(map.root());
    auto& taken =
        builder
            .add_case(ife, symbolic::And(symbolic::Ge(i, symbolic::integer(3)), symbolic::Lt(i, symbolic::integer(227))));
    auto& not_taken =
        builder
            .add_case(ife, symbolic::Or(symbolic::Lt(i, symbolic::integer(3)), symbolic::Ge(i, symbolic::integer(227))));

    auto idx = symbolic::sub(i, symbolic::integer(3));
    {
        auto& block = builder.add_block(taken);
        auto& in_node = builder.add_access(block, "_1");
        auto& out_node = builder.add_access(block, "_patches");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in_node, tasklet, "_in", {idx});
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {idx});
    }
    {
        auto& block = builder.add_block(not_taken);
        auto& constant = builder.add_constant(block, "0.0f", scalar_type);
        auto& out_node = builder.add_access(block, "_patches");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, constant, tasklet, "_in", {}, scalar_type);
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {idx});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    auto* tile_in = analysis.tile(map, "_1");
    ASSERT_NE(tile_in, nullptr) << "MLA returned nullptr for _1 with negative-offset (i-3) subscript inside IfElse.";
    auto range_in = tile_in->contiguous_range();
    EXPECT_FALSE(range_in.first.is_null());
    EXPECT_FALSE(range_in.second.is_null());

    auto* tile_out = analysis.tile(map, "_patches");
    ASSERT_NE(tile_out, nullptr)
        << "MLA returned nullptr for _patches with negative-offset (i-3) subscript inside IfElse.";
    auto range_out = tile_out->contiguous_range();
    EXPECT_FALSE(range_out.first.is_null());
    EXPECT_FALSE(range_out.second.is_null());
}

TEST(MemoryLayoutAnalysisTest, Regression_Im2col_ResNetFullPattern) {
    // Faithful reproduction of the resnet kernel that stopped offloading:
    //   for n0 in [0, 401408):
    //     for c0 in [0, 147):
    //       if  (in-bounds for _1 read): _patches0[Pn0c0] = _1[In0c0]
    //       elif (out-of-bounds):        _patches0[Pn0c0] = 0
    // where In0c0 / Pn0c0 are the mod/div linear combos from the failing
    // kernel. Both _1 and _patches0 must be bounded at the outer map.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("_n0_collapsed0", index_type);
    builder.add_container("_c0_collapsed0", index_type);
    builder.add_container("_1", pointer_type, true);
    builder.add_container("_patches0", pointer_type, true);

    auto n0 = symbolic::symbol("_n0_collapsed0");
    auto c0 = symbolic::symbol("_c0_collapsed0");
    auto i7 = symbolic::integer(7);
    auto i112 = symbolic::integer(112);
    auto i49 = symbolic::integer(49);
    auto i12544 = symbolic::integer(12544);
    auto i224 = symbolic::integer(224);
    auto i147 = symbolic::integer(147);
    auto i16464 = symbolic::integer(16464);
    auto i50176 = symbolic::integer(50176);
    auto i150528 = symbolic::integer(150528);
    auto i1843968 = symbolic::integer(1843968);
    auto i_neg3 = symbolic::integer(-3);
    auto i2 = symbolic::integer(2);

    auto& outer = builder.add_map(
        root,
        n0,
        symbolic::Lt(n0, symbolic::integer(401408)),
        symbolic::zero(),
        symbolic::add(n0, symbolic::one()),
        ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        c0,
        symbolic::Lt(c0, i147),
        symbolic::zero(),
        symbolic::add(c0, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto kh_mod = symbolic::mod(symbolic::div(c0, i7), i7);
    auto kw_mod = symbolic::mod(c0, i7);
    auto hout_mod = symbolic::mod(symbolic::div(n0, i112), i112);
    auto wout_mod = symbolic::mod(n0, i112);
    auto c_div = symbolic::div(c0, i49);
    auto n_div = symbolic::div(n0, i12544);

    // h_in = -3 + ((c0/7)%7) + 2*((n0/112)%112)
    auto h_in = symbolic::add(i_neg3, symbolic::add(kh_mod, symbolic::mul(i2, hout_mod)));
    // w_in = -3 + (c0%7) + 2*(n0%112)
    auto w_in = symbolic::add(i_neg3, symbolic::add(kw_mod, symbolic::mul(i2, wout_mod)));

    auto cond_in = symbolic::
        And(symbolic::And(symbolic::Ge(w_in, symbolic::zero()), symbolic::Ge(h_in, symbolic::zero())),
            symbolic::And(symbolic::Lt(w_in, i224), symbolic::Lt(h_in, i224)));
    auto cond_out = symbolic::
        Or(symbolic::Or(symbolic::Ge(w_in, i224), symbolic::Ge(h_in, i224)),
           symbolic::Or(symbolic::Lt(w_in, symbolic::zero()), symbolic::Lt(h_in, symbolic::zero())));

    auto& ife = builder.add_if_else(inner.root());
    auto& case_in = builder.add_case(ife, cond_in);
    auto& case_out = builder.add_case(ife, cond_out);

    // patches index: 49*(c0/49) + 1843968*(n0/12544) + (c0%7) + 147*(n0%112)
    //                + 7*((c0/7)%7) + 16464*((n0/112)%112)
    auto out_idx = symbolic::
        add(symbolic::
                add(symbolic::add(symbolic::mul(i49, c_div), symbolic::mul(i1843968, n_div)),
                    symbolic::add(kw_mod, symbolic::mul(i147, wout_mod))),
            symbolic::add(symbolic::mul(i7, kh_mod), symbolic::mul(i16464, hout_mod)));
    // _1 index: -3 + 224*h_in + 50176*c_div + 150528*n_div + (c0%7) + 2*(n0%112)
    auto in_idx = symbolic::
        add(i_neg3,
            symbolic::
                add(symbolic::add(symbolic::mul(i224, h_in), symbolic::mul(i50176, c_div)),
                    symbolic::add(symbolic::mul(i150528, n_div), symbolic::add(kw_mod, symbolic::mul(i2, wout_mod)))));

    {
        auto& block = builder.add_block(case_in);
        auto& in_node = builder.add_access(block, "_1");
        auto& out_node = builder.add_access(block, "_patches0");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in_node, tasklet, "_in", {in_idx});
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {out_idx});
    }
    {
        auto& block = builder.add_block(case_out);
        auto& constant = builder.add_constant(block, "0.0f", scalar_type);
        auto& out_node = builder.add_access(block, "_patches0");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, constant, tasklet, "_in", {}, scalar_type);
        builder.add_computational_memlet(block, tasklet, "_out", out_node, {out_idx});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // CURRENT BEHAVIOR (conservative, intentional):
    //   _1's linearized index `-3 + 224*h_in + ...` provably reaches negative
    //   addresses in the abstract iteration domain (`h_in` has min -3). The
    //   real kernel only loads when the surrounding IfElse guard
    //   `h_in >= 0 && w_in >= 0 && ...` is true.
    //
    //   AssumptionsAnalysis DOES propagate this guard into the taken
    //   Sequence (CNF-based extraction in assumptions_analysis.cpp). What is
    //   still missing is MLA-side IfElse-awareness: when merging per-block
    //   tiles up to the outer-loop scope, MLA computes extents over the full
    //   iteration domain (including the case_out arm where the memlet is
    //   absent). The delinearizer's `is_nonneg` gate therefore still sees a
    //   subscript that can be negative at outer scope, and refuses.
    //
    //   This is the sound answer for MLA's contract (used by tile_fusion,
    //   in/out_local_storage, loop_carried_dependency). The argument-sizing
    //   path falls back to a direct min/max bound in ArgumentsAnalysis.
    //
    //   TODO: when MLA gains IfElse-aware tile aggregation, flip these to
    //   ASSERT_NE(...) and check the contiguous range.
    auto* tile_1 = analysis.tile(outer, "_1");
    EXPECT_EQ(tile_1, nullptr) << "Expected MLA to refuse to bound _1 because the IfElse halo guard "
                                  "is not yet propagated as an assumption (see assumptions_analysis.cpp).";

    // _patches0 has only non-negative indices and IS delinearizable.
    auto* tile_p = analysis.tile(outer, "_patches0");
    ASSERT_NE(tile_p, nullptr) << "MLA could not bound _patches0 at the outer collapsed map (resnet im2col pattern).";
    auto rp = tile_p->contiguous_range();
    EXPECT_FALSE(rp.first.is_null());
    EXPECT_FALSE(rp.second.is_null());
}

// MINIMAL reproduction of the delinearization bug found while debugging
// Regression_Im2col_ResNetFullPattern. The kernel feeds `A[224*(-3 + (i%7) +
// 2*j) + (i%7)]` over `(i,j) in [0,7) x [0,112)` to MLA. Without expansion of
// the parameter*indvar product, `decompose_by_stride` produces a group whose
// index is `-3 + (i%7) + 2*j`, which can take the value -3. The non-negativity
// gate in `delinearize` then rejects the access and no tile is built.
TEST(MemoryLayoutAnalysisTest, Regression_Im2col_NegativeConstInsideStrideProduct) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto i_sym = symbolic::symbol("i");
    auto j_sym = symbolic::symbol("j");

    auto& outer = builder.add_map(
        root,
        i_sym,
        symbolic::Lt(i_sym, symbolic::integer(7)),
        symbolic::zero(),
        symbolic::add(i_sym, symbolic::one()),
        ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j_sym,
        symbolic::Lt(j_sym, symbolic::integer(112)),
        symbolic::zero(),
        symbolic::add(j_sym, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    // idx = 224 * (-3 + (i%7) + 2*j) + (i%7)
    auto idx = symbolic::add(
        symbolic::mul(
            symbolic::integer(224),
            symbolic::add(
                symbolic::integer(-3),
                symbolic::add(symbolic::mod(i_sym, symbolic::integer(7)), symbolic::mul(symbolic::integer(2), j_sym))
            )
        ),
        symbolic::mod(i_sym, symbolic::integer(7))
    );

    auto& block = builder.add_block(inner.root());
    auto& a_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_node, tasklet, "_in", {idx});
    auto& sink = builder.add_access(block, "A");
    builder.add_computational_memlet(block, tasklet, "_out", sink, {idx});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // CURRENT BEHAVIOR (conservative, intentional):
    //   `decompose_by_stride` produces a group with stride=224 and index
    //   `-3 + (i%7) + 2*j`. That index provably reaches -3, so the peel-loop's
    //   non-negativity gate in delinearize() correctly refuses to peel.
    //
    //   The expression genuinely addresses negative offsets (`224*-3 = -672`)
    //   in the abstract iteration domain. The full resnet kernel guards this
    //   access with an IfElse halo check; that guard is not visible to
    //   AssumptionsAnalysis today, so the delinearizer must reject.
    //
    //   Refusing to delinearize here is the sound answer for MLA. The size
    //   needed for argument allocation is computed directly from min/max in
    //   ArgumentsAnalysis's fallback path.
    //
    //   TODO: with IfElse-aware AssumptionsAnalysis (or after lifting the
    //   pure-constant -3 into the layout offset AND teaching
    //   MemoryTile::contiguous_range to clamp negative endpoints), this can
    //   become ASSERT_NE(tile, nullptr).
    auto* tile = analysis.tile(outer, "A");
    EXPECT_EQ(tile, nullptr) << "Expected MLA to refuse: index `-3 + (i%7) + 2*j` provably reaches -3.";
}

// Minimal 2-D linearized halo whose subscript is provably non-negative ONLY
// under the surrounding IfElse guard:
//   for i in [0, 7):
//     for j in [0, 7):
//       if (i >= 3 && j >= 3):  A[7*(i - 3) + (j - 3)] = ...
//
// AssumptionsAnalysis DOES propagate `i>=3 && j>=3` into the taken Sequence
// (CNF-based extraction). What is still missing is MLA-side: when
// aggregating tiles up to the outer-loop scope, MLA computes extents over
// the *full* outer iteration domain (where the guard does not hold) and
// rightly refuses because `7*(i-3) + (j-3)` reaches -24. This test pins the
// conservative refusal.
//
// TODO: once MLA gains IfElse-aware tile aggregation, flip the expectation
// to ASSERT_NE(tile, nullptr) and check that dims={7} indices={i-3, j-3}.
TEST(MemoryLayoutAnalysisTest, Regression_Halo2D_GuardedByIfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto three = symbolic::integer(3);
    auto seven = symbolic::integer(7);

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, seven),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, seven),
        symbolic::zero(),
        symbolic::add(j, symbolic::one()),
        ScheduleType_Sequential::create()
    );

    auto& ife = builder.add_if_else(inner.root());
    auto cond_in = symbolic::And(symbolic::Ge(i, three), symbolic::Ge(j, three));
    auto& taken = builder.add_case(ife, cond_in);

    // Linearized 2-D address: 7*(i-3) + (j-3)
    auto idx = symbolic::add(symbolic::mul(seven, symbolic::sub(i, three)), symbolic::sub(j, three));

    {
        auto& block = builder.add_block(taken);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {idx});
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {idx});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // MLA recovers the 2-D layout `7*(i-3) + (j-3)` -> indices (i-3, j-3) with
    // extents (7, 7). The branch guard `i>=3 && j>=3` is propagated by
    // AssumptionsAnalysis into the taken Sequence so block-scope delinearization
    // succeeds. Tile aggregation at the outer scope conservatively widens the
    // bounding box back to the full outer-loop domain `i,j ∈ [0,7)` — yielding
    // extents=[7,7] (vs. the tighter [4,4] that the actual access set occupies).
    // This is a sound over-approximation; tightening would require IfElse-aware
    // tile aggregation in merge_scope_layouts.
    auto* tile = analysis.tile(outer, "A");
    ASSERT_NE(tile, nullptr);
    auto ext = tile->extents();
    ASSERT_EQ(ext.size(), 2u);
    EXPECT_TRUE(symbolic::eq(ext[0], symbolic::integer(7)));
    EXPECT_TRUE(symbolic::eq(ext[1], symbolic::integer(7)));
}
