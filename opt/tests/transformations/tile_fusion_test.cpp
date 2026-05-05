#include "sdfg/transformations/tile_fusion.h"

#include <gtest/gtest.h>

#include <set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

/**
 * Helper: Build a Jacobi-1D-like SDFG with two Maps inside a For time loop.
 *
 *   for t = 0..TSTEPS:
 *     map i = 0..N-2: B[1+i] = 0.333*(A[i] + A[1+i] + A[2+i])
 *     map j = 0..N-2: A[1+j] = 0.333*(B[j] + B[1+j] + B[2+j])
 *
 * Returns the builder (not moved), along with references to the two maps.
 */
struct Jacobi1DFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;
    structured_control_flow::Map* map_k1 = nullptr;
    structured_control_flow::Map* map_k2 = nullptr;
    structured_control_flow::StructuredLoop* loop_t = nullptr;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("jacobi_1d", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("TSTEPS", sym_desc, true);
        builder->add_container("N", sym_desc, true);
        builder->add_container("t", sym_desc);
        builder->add_container("i", sym_desc);
        builder->add_container("j", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp1", elem_desc);
        builder->add_container("tmp2", elem_desc);
        builder->add_container("tmp3", elem_desc);
        builder->add_container("tmp4", elem_desc);

        types::Array desc_1d(elem_desc, symbolic::symbol("N"));
        builder->add_container("A", desc_1d, true);
        builder->add_container("B", desc_1d, true);

        auto& root = builder->subject().root();

        // Time loop
        auto& time_loop = builder->add_for(
            root,
            symbolic::symbol("t"),
            symbolic::Lt(symbolic::symbol("t"), symbolic::symbol("TSTEPS")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("t"), symbolic::integer(1))
        );
        loop_t = &time_loop;
        auto& body_t = time_loop.root();

        // K1: map i = 0..N-2: B[1+i] = 0.333 * (A[i] + A[1+i] + A[2+i])
        auto& k1 = builder->add_map(
            body_t,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k1 = &k1;
        {
            // Block 1: tmp1 = A[i] + A[1+i]
            auto& block1 = builder->add_block(k1.root());
            auto& a_in1 = builder->add_access(block1, "A");
            auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp1_out = builder->add_access(block1, "tmp1");
            builder->add_computational_memlet(block1, a_in1, tasklet1, "_in1", {symbolic::symbol("i")}, desc_1d);
            builder->add_computational_memlet(
                block1, a_in1, tasklet1, "_in2", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, desc_1d
            );
            builder->add_computational_memlet(block1, tasklet1, "_out", tmp1_out, {});

            // Block 2: tmp2 = tmp1 + A[2+i]
            auto& block2 = builder->add_block(k1.root());
            auto& tmp1_in = builder->add_access(block2, "tmp1");
            auto& a_in2 = builder->add_access(block2, "A");
            auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp2_out = builder->add_access(block2, "tmp2");
            builder->add_computational_memlet(block2, tmp1_in, tasklet2, "_in1", {});
            builder->add_computational_memlet(
                block2, a_in2, tasklet2, "_in2", {symbolic::add(symbolic::symbol("i"), symbolic::integer(2))}, desc_1d
            );
            builder->add_computational_memlet(block2, tasklet2, "_out", tmp2_out, {});

            // Block 3: B[1+i] = 0.333 * tmp2
            auto& block3 = builder->add_block(k1.root());
            auto& tmp2_in = builder->add_access(block3, "tmp2");
            auto& const_node = builder->add_constant(block3, "0.333", elem_desc);
            auto& tasklet3 = builder->add_tasklet(block3, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& b_out = builder->add_access(block3, "B");
            builder->add_computational_memlet(block3, const_node, tasklet3, "_in1", {});
            builder->add_computational_memlet(block3, tmp2_in, tasklet3, "_in2", {});
            builder->add_computational_memlet(
                block3, tasklet3, "_out", b_out, {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, desc_1d
            );
        }

        // K2: map j = 0..N-2: A[1+j] = 0.333 * (B[j] + B[1+j] + B[2+j])
        auto& k2 = builder->add_map(
            body_t,
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k2 = &k2;
        {
            // Block 1: tmp3 = B[j] + B[1+j]
            auto& block1 = builder->add_block(k2.root());
            auto& b_in1 = builder->add_access(block1, "B");
            auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp3_out = builder->add_access(block1, "tmp3");
            builder->add_computational_memlet(block1, b_in1, tasklet1, "_in1", {symbolic::symbol("j")}, desc_1d);
            builder->add_computational_memlet(
                block1, b_in1, tasklet1, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, desc_1d
            );
            builder->add_computational_memlet(block1, tasklet1, "_out", tmp3_out, {});

            // Block 2: tmp4 = tmp3 + B[2+j]
            auto& block2 = builder->add_block(k2.root());
            auto& tmp3_in = builder->add_access(block2, "tmp3");
            auto& b_in2 = builder->add_access(block2, "B");
            auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp4_out = builder->add_access(block2, "tmp4");
            builder->add_computational_memlet(block2, tmp3_in, tasklet2, "_in1", {});
            builder->add_computational_memlet(
                block2, b_in2, tasklet2, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(2))}, desc_1d
            );
            builder->add_computational_memlet(block2, tasklet2, "_out", tmp4_out, {});

            // Block 3: A[1+j] = 0.333 * tmp4
            auto& block3 = builder->add_block(k2.root());
            auto& tmp4_in = builder->add_access(block3, "tmp4");
            auto& const_node = builder->add_constant(block3, "0.333", elem_desc);
            auto& tasklet3 = builder->add_tasklet(block3, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& a_out = builder->add_access(block3, "A");
            builder->add_computational_memlet(block3, const_node, tasklet3, "_in1", {});
            builder->add_computational_memlet(block3, tmp4_in, tasklet3, "_in2", {});
            builder->add_computational_memlet(
                block3, tasklet3, "_out", a_out, {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, desc_1d
            );
        }
    }
};

TEST(TileFusionTest, Jacobi1D_Basic) {
    Jacobi1DFixture fixture;
    fixture.build();
    auto& builder = *fixture.builder;

    // First, tile both inner maps with tile size 32
    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Find the maps inside the time loop
    auto& time_loop_body = fixture.loop_t->root();
    // After move, we need to re-navigate the SDFG
    auto& root = builder_opt.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* time_loop = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(time_loop, nullptr);
    ASSERT_EQ(time_loop->root().size(), 2);

    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // Tile K1
    transformations::LoopTiling tiling_k1(*k1, 32);
    ASSERT_TRUE(tiling_k1.can_be_applied(builder_opt, analysis_manager));
    tiling_k1.apply(builder_opt, analysis_manager);
    auto* k1_outer = dynamic_cast<structured_control_flow::Map*>(tiling_k1.outer_loop());
    auto* k1_inner = dynamic_cast<structured_control_flow::Map*>(tiling_k1.inner_loop());
    ASSERT_NE(k1_outer, nullptr);
    ASSERT_NE(k1_inner, nullptr);

    // Tile K2
    transformations::LoopTiling tiling_k2(*k2, 32);
    ASSERT_TRUE(tiling_k2.can_be_applied(builder_opt, analysis_manager));
    tiling_k2.apply(builder_opt, analysis_manager);
    auto* k2_outer = dynamic_cast<structured_control_flow::Map*>(tiling_k2.outer_loop());
    auto* k2_inner = dynamic_cast<structured_control_flow::Map*>(tiling_k2.inner_loop());
    ASSERT_NE(k2_outer, nullptr);
    ASSERT_NE(k2_inner, nullptr);

    // Cleanup
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    // Now apply TileFusion on the two tiled outer maps
    ASSERT_EQ(time_loop->root().size(), 2);
    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);
    ASSERT_NE(tile_k1, nullptr);
    ASSERT_NE(tile_k2, nullptr);

    transformations::TileFusion tile_fusion(*tile_k1, *tile_k2);
    EXPECT_TRUE(tile_fusion.can_be_applied(builder_opt, analysis_manager));
    EXPECT_EQ(tile_fusion.radius(), 1);

    tile_fusion.apply(builder_opt, analysis_manager);

    // Verify structure: time loop now has 1 child (fused For tile loop).
    // The init copy is inside the fused loop guarded by if (tile == init).
    ASSERT_EQ(time_loop->root().size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&time_loop->root().at(0).first);
    ASSERT_NE(fused_tile, nullptr);

    // The fused tile loop should have 5 children with double-buffer:
    //   1. if-else guard (init copy on first tile iteration)
    //   2. pre-fetch copy (buf_pf = C[next tile])
    //   3. producer Map (K1, reads buf_cur)
    //   4. consumer Map (K2, writes original C)
    //   5. swap copy (buf_cur = buf_pf)
    ASSERT_EQ(fused_tile->root().size(), 5);
    auto* init_if_else = dynamic_cast<structured_control_flow::IfElse*>(&fused_tile->root().at(0).first);
    auto* pf_copy = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(1).first);
    auto* fused_producer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(2).first);
    auto* fused_consumer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(3).first);
    auto* swap_copy = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(4).first);
    ASSERT_NE(init_if_else, nullptr);
    ASSERT_NE(pf_copy, nullptr);
    ASSERT_NE(fused_producer, nullptr);
    ASSERT_NE(fused_consumer, nullptr);
    ASSERT_NE(swap_copy, nullptr);

    // Producer should have extended bounds (radius=1 for Jacobi stencil)
    // Init should involve max(0, tile - 1)
    EXPECT_TRUE(symbolic::uses(fused_producer->init(), fused_tile->indvar()->get_name()));

    // Consumer bounds should reference the tile indvar
    EXPECT_TRUE(symbolic::uses(fused_consumer->init(), fused_tile->indvar()->get_name()));

    // Both inner maps should preserve their computation blocks
    EXPECT_GE(fused_producer->root().size(), 1);
    EXPECT_GE(fused_consumer->root().size(), 1);
}

TEST(TileFusionTest, NonConsecutiveMaps_ShouldFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);
    builder.add_container("C", desc_1d, true);

    auto& root = builder.subject().root();

    // Map 1
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        // Inner map
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block, "A");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // Blocker map (inserted between m1 and m2)
    auto& blocker = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(blocker.root());
        auto& a_in = builder.add_access(block, "A");
        auto& c_out = builder.add_access(block, "C");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("k")}, desc_1d);
    }

    // Map 2
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& b_in = builder.add_access(block, "B");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    EXPECT_FALSE(tile_fusion.can_be_applied(builder, analysis_manager));
}

TEST(TileFusionTest, IncompatibleTileSizes_ShouldFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);

    auto& root = builder.subject().root();

    // Map 1 with tile size 32
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block, "A");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // Map 2 with tile size 64 (different!)
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(64)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& b_in = builder.add_access(block, "B");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    EXPECT_FALSE(tile_fusion.can_be_applied(builder, analysis_manager));
}

TEST(TileFusionTest, NoSharedContainer_ShouldFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);
    builder.add_container("C", desc_1d, true);
    builder.add_container("D", desc_1d, true);

    auto& root = builder.subject().root();

    // Map 1: reads A, writes B
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block, "A");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // Map 2: reads C, writes D — NO shared container with Map 1
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& c_in = builder.add_access(block, "C");
        auto& d_out = builder.add_access(block, "D");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, c_in, tasklet, "_in", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", d_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    EXPECT_FALSE(tile_fusion.can_be_applied(builder, analysis_manager));
}

TEST(TileFusionTest, ConsumerAlsoWritesIntermediate_ShouldFail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);

    auto& root = builder.subject().root();

    // Map 1: reads A, writes B
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block, "A");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // Map 2: reads B AND writes B (consumer also produces to intermediate)
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& b_in = builder.add_access(block, "B");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    EXPECT_FALSE(tile_fusion.can_be_applied(builder, analysis_manager));
}

TEST(TileFusionTest, ZeroRadiusElementwise) {
    // Elementwise: B[i] = f(A[i]), A[j] = g(B[j])
    // Radius should be 0, but transformation should still work
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);

    auto& root = builder.subject().root();

    // Map 1: B[i] = A[i] (elementwise, radius 0)
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::
                And(symbolic::Lt(symbolic::symbol("i"), symbolic::add(m1.indvar(), symbolic::integer(32))),
                    symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"))),
            m1.indvar(),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block, "A");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // Map 2: A[j] = B[j] (elementwise, radius 0)
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::
                And(symbolic::Lt(symbolic::symbol("j"), symbolic::add(m2.indvar(), symbolic::integer(32))),
                    symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N"))),
            m2.indvar(),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block = builder.add_block(inner.root());
        auto& b_in = builder.add_access(block, "B");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    EXPECT_TRUE(tile_fusion.can_be_applied(builder, analysis_manager));
    EXPECT_EQ(tile_fusion.radius(), 0);

    tile_fusion.apply(builder, analysis_manager);

    // Verify: single For tile loop with 2 inner Maps
    auto& new_root = builder.subject().root();
    ASSERT_EQ(new_root.size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    ASSERT_NE(fused_tile, nullptr);
    ASSERT_EQ(fused_tile->root().size(), 2);

    auto* producer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(0).first);
    auto* consumer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(1).first);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);
}

// =============================================================================
// Double-buffer pre-fetch tests
// =============================================================================

/**
 * Verify that the Jacobi-1D double-buffer creates the correct buffer containers
 * and that K1's reads are redirected to buf_cur.
 */
TEST(TileFusionTest, Jacobi1D_DoubleBufferContainers) {
    Jacobi1DFixture fixture;
    fixture.build();
    auto& builder = *fixture.builder;

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& root = builder_opt.subject().root();
    auto* time_loop = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    transformations::LoopTiling tiling_k1(*k1, 32);
    ASSERT_TRUE(tiling_k1.can_be_applied(builder_opt, analysis_manager));
    tiling_k1.apply(builder_opt, analysis_manager);

    transformations::LoopTiling tiling_k2(*k2, 32);
    ASSERT_TRUE(tiling_k2.can_be_applied(builder_opt, analysis_manager));
    tiling_k2.apply(builder_opt, analysis_manager);

    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    transformations::TileFusion tile_fusion(*tile_k1, *tile_k2);
    ASSERT_TRUE(tile_fusion.can_be_applied(builder_opt, analysis_manager));
    tile_fusion.apply(builder_opt, analysis_manager);

    auto& sdfg = builder_opt.subject();

    // Verify buffer containers were created
    EXPECT_NO_THROW(sdfg.type("__tf_buf_cur_A"));
    EXPECT_NO_THROW(sdfg.type("__tf_buf_pf_A"));

    // Verify buffer type is Array of Double with correct size
    // Buffer size = tile_size + 2*radius + max_offset - min_offset = 32 + 2*1 + 2 - 0 = 36
    auto& buf_cur_type = sdfg.type("__tf_buf_cur_A");
    auto* buf_arr = dynamic_cast<const types::Array*>(&buf_cur_type);
    ASSERT_NE(buf_arr, nullptr);
    auto& elem_type = buf_arr->element_type();
    auto* elem_scalar = dynamic_cast<const types::Scalar*>(&elem_type);
    ASSERT_NE(elem_scalar, nullptr);
    EXPECT_EQ(elem_scalar->primitive_type(), types::PrimitiveType::Double);

    // Verify K1 reads from buf_cur, not from A
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&time_loop->root().at(0).first);
    auto* fused_producer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(2).first);

    // Walk K1's blocks and check that reads of "A" are now reads of "__tf_buf_cur_A"
    bool found_buf_cur_read = false;
    bool found_direct_a_read = false;
    std::function<void(structured_control_flow::ControlFlowNode&)> check_reads;
    check_reads = [&](structured_control_flow::ControlFlowNode& node) {
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
            for (auto* access : block->dataflow().data_nodes()) {
                if (access->data() == "__tf_buf_cur_A" && block->dataflow().out_degree(*access) > 0) {
                    found_buf_cur_read = true;
                }
                if (access->data() == "A" && block->dataflow().out_degree(*access) > 0) {
                    found_direct_a_read = true;
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                check_reads(seq->at(i).first);
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
            check_reads(loop->root());
        }
    };
    check_reads(fused_producer->root());

    EXPECT_TRUE(found_buf_cur_read) << "K1 should read from __tf_buf_cur_A";
    EXPECT_FALSE(found_direct_a_read) << "K1 should NOT read directly from A";

    // Verify K2 still writes to A (not to a buffer)
    auto* fused_consumer = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(3).first);
    bool found_a_write = false;
    std::function<void(structured_control_flow::ControlFlowNode&)> check_writes;
    check_writes = [&](structured_control_flow::ControlFlowNode& node) {
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
            for (auto* access : block->dataflow().data_nodes()) {
                if (access->data() == "A" && block->dataflow().in_degree(*access) > 0) {
                    found_a_write = true;
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                check_writes(seq->at(i).first);
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
            check_writes(loop->root());
        }
    };
    check_writes(fused_consumer->root());
    EXPECT_TRUE(found_a_write) << "K2 should still write to A";
}

/**
 * Verify that the init copy, pre-fetch, and swap loops all copy the right
 * container and reference the right buffers.
 */
TEST(TileFusionTest, Jacobi1D_CopyLoopTargets) {
    Jacobi1DFixture fixture;
    fixture.build();
    auto& builder = *fixture.builder;

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& root = builder_opt.subject().root();
    auto* time_loop = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    transformations::LoopTiling tiling_k1(*k1, 32);
    tiling_k1.can_be_applied(builder_opt, analysis_manager);
    tiling_k1.apply(builder_opt, analysis_manager);
    transformations::LoopTiling tiling_k2(*k2, 32);
    tiling_k2.can_be_applied(builder_opt, analysis_manager);
    tiling_k2.apply(builder_opt, analysis_manager);

    passes::SequenceFusion sf;
    passes::DeadCFGElimination dcfg;
    bool a;
    do {
        a = false;
        a |= dcfg.run(builder_opt, analysis_manager);
        a |= sf.run(builder_opt, analysis_manager);
    } while (a);

    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    transformations::TileFusion tile_fusion(*tile_k1, *tile_k2);
    tile_fusion.can_be_applied(builder_opt, analysis_manager);
    tile_fusion.apply(builder_opt, analysis_manager);

    // Helper: collect data node names from a Map's body
    auto collect_data_nodes = [](structured_control_flow::Map& map) {
        std::set<std::string> names;
        auto& seq = map.root();
        for (size_t i = 0; i < seq.size(); i++) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&seq.at(i).first);
            if (!block) continue;
            for (auto* access : block->dataflow().data_nodes()) {
                names.insert(access->data());
            }
        }
        return names;
    };

    // Init copy: inside if-else at fused_for[0], reads A, writes __tf_buf_cur_A
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&time_loop->root().at(0).first);
    ASSERT_NE(fused_tile, nullptr);

    auto* init_if_else = dynamic_cast<structured_control_flow::IfElse*>(&fused_tile->root().at(0).first);
    ASSERT_NE(init_if_else, nullptr);
    ASSERT_GE(init_if_else->size(), 1);
    auto& init_then = init_if_else->at(0).first;
    ASSERT_GE(init_then.size(), 1);
    auto* init_copy = dynamic_cast<structured_control_flow::Map*>(&init_then.at(0).first);
    ASSERT_NE(init_copy, nullptr);
    auto init_nodes = collect_data_nodes(*init_copy);
    EXPECT_TRUE(init_nodes.count("A")) << "Init copy should read from A";
    EXPECT_TRUE(init_nodes.count("__tf_buf_cur_A")) << "Init copy should write to __tf_buf_cur_A";

    // Pre-fetch: reads A, writes __tf_buf_pf_A
    auto* pf_copy = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(1).first);
    auto pf_nodes = collect_data_nodes(*pf_copy);
    EXPECT_TRUE(pf_nodes.count("A")) << "Pre-fetch should read from A";
    EXPECT_TRUE(pf_nodes.count("__tf_buf_pf_A")) << "Pre-fetch should write to __tf_buf_pf_A";

    // Swap: reads __tf_buf_pf_A, writes __tf_buf_cur_A
    auto* swap_copy = dynamic_cast<structured_control_flow::Map*>(&fused_tile->root().at(4).first);
    auto swap_nodes = collect_data_nodes(*swap_copy);
    EXPECT_TRUE(swap_nodes.count("__tf_buf_pf_A")) << "Swap should read from __tf_buf_pf_A";
    EXPECT_TRUE(swap_nodes.count("__tf_buf_cur_A")) << "Swap should write to __tf_buf_cur_A";
}

/**
 * When K1 writes B and K2 reads B, but K2 does NOT write something K1 reads,
 * no double-buffer should be created.
 */
TEST(TileFusionTest, NoCyclicDependency_NoBuffers) {
    // K1: B[i] = A[i] + A[1+i] + A[2+i]  (reads A, writes B)
    // K2: C[j] = B[j] + B[1+j] + B[2+j]  (reads B, writes C)
    // No cyclic: K2 writes C, K1 doesn't read C.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("tmp1", elem_desc);
    types::Array desc_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", desc_1d, true);
    builder.add_container("B", desc_1d, true);
    builder.add_container("C", desc_1d, true);

    auto& root = builder.subject().root();

    // K1: B[i] = A[i] + A[1+i] (stencil, radius=1)
    auto& m1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m1.root(),
            symbolic::symbol("i"),
            symbolic::
                And(symbolic::Lt(symbolic::symbol("i"), symbolic::add(m1.indvar(), symbolic::integer(32))),
                    symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)))),
            m1.indvar(),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block1 = builder.add_block(inner.root());
        auto& a_in = builder.add_access(block1, "A");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        auto& b_out = builder.add_access(block1, "B");
        builder.add_computational_memlet(block1, a_in, tasklet, "_in1", {symbolic::symbol("i")}, desc_1d);
        builder.add_computational_memlet(
            block1, a_in, tasklet, "_in2", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, desc_1d
        );
        builder.add_computational_memlet(block1, tasklet, "_out", b_out, {symbolic::symbol("i")}, desc_1d);
    }

    // K2: C[j] = B[j] + B[1+j] (stencil, radius=1)
    auto& m2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(32)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& inner = builder.add_map(
            m2.root(),
            symbolic::symbol("j"),
            symbolic::
                And(symbolic::Lt(symbolic::symbol("j"), symbolic::add(m2.indvar(), symbolic::integer(32))),
                    symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1)))),
            m2.indvar(),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& block1 = builder.add_block(inner.root());
        auto& b_in = builder.add_access(block1, "B");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        auto& c_out = builder.add_access(block1, "C");
        builder.add_computational_memlet(block1, b_in, tasklet, "_in1", {symbolic::symbol("j")}, desc_1d);
        builder.add_computational_memlet(
            block1, b_in, tasklet, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, desc_1d
        );
        builder.add_computational_memlet(block1, tasklet, "_out", c_out, {symbolic::symbol("j")}, desc_1d);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TileFusion tile_fusion(m1, m2);
    ASSERT_TRUE(tile_fusion.can_be_applied(builder, analysis_manager));
    EXPECT_EQ(tile_fusion.radius(), 1);

    tile_fusion.apply(builder, analysis_manager);

    // No cyclic dependency → fused For has 2 children (K1 + K2), no buffer loops
    ASSERT_EQ(root.size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(fused_tile, nullptr);
    ASSERT_EQ(fused_tile->root().size(), 2) << "Without cyclic deps, no buffer loops should be added";

    // Verify no buffer containers were created
    EXPECT_THROW(builder.subject().type("__tf_buf_cur_A"), std::exception);
    EXPECT_THROW(builder.subject().type("__tf_buf_pf_A"), std::exception);
}

/**
 * Verify TileFusion serialization round-trip still works with the new fields.
 */
TEST(TileFusionTest, Jacobi1D_Serialization) {
    Jacobi1DFixture fixture;
    fixture.build();
    auto& builder = *fixture.builder;

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& root = builder_opt.subject().root();
    auto* time_loop = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    transformations::LoopTiling tiling_k1(*k1, 32);
    tiling_k1.can_be_applied(builder_opt, analysis_manager);
    tiling_k1.apply(builder_opt, analysis_manager);
    transformations::LoopTiling tiling_k2(*k2, 32);
    tiling_k2.can_be_applied(builder_opt, analysis_manager);
    tiling_k2.apply(builder_opt, analysis_manager);

    passes::SequenceFusion sf;
    passes::DeadCFGElimination dcfg;
    bool a;
    do {
        a = false;
        a |= dcfg.run(builder_opt, analysis_manager);
        a |= sf.run(builder_opt, analysis_manager);
    } while (a);

    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&time_loop->root().at(1).first);

    // Serialize before applying
    transformations::TileFusion tile_fusion(*tile_k1, *tile_k2);
    ASSERT_TRUE(tile_fusion.can_be_applied(builder_opt, analysis_manager));

    nlohmann::json j;
    tile_fusion.to_json(j);

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "TileFusion");
    EXPECT_EQ(j["parameters"]["radius"], 1);
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_TRUE(j["subgraph"].contains("1"));

    // Round-trip: from_json should produce a valid transformation
    auto tile_fusion2 = transformations::TileFusion::from_json(builder_opt, j);
    EXPECT_TRUE(tile_fusion2.can_be_applied(builder_opt, analysis_manager));
    EXPECT_EQ(tile_fusion2.radius(), 1);
}
