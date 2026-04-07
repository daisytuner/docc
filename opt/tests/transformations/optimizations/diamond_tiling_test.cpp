#include <gtest/gtest.h>
#include <sdfg/transformations/recorder.h>

#include <fstream>
#include <memory>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_shift.h"
#include "sdfg/transformations/loop_skewing.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/tile_fusion.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

/**
 * Builds a Jacobi-1D SDFG with two Maps inside a For time loop:
 *
 *   for t = 0..TSTEPS:
 *     map i = 0..N-2: B[1+i] = 0.333*(A[i] + A[1+i] + A[2+i])
 *     map j = 0..N-2: A[1+j] = 0.333*(B[j] + B[1+j] + B[2+j])
 *
 * Diamond tiling pipeline:
 *   1. Tile(K1, 32) + Tile(K2, 32)
 *   2. Cleanup passes (DCE + SequenceFusion)
 *   3. TileFusion(K1_tile, K2_tile) — fuse into single tile loop, radius=1
 *   4. LoopSkewing(t, tile, factor=32) — shift tile by 32*t
 *   5. LoopInterchange(t, tile) — FM with α=32
 *
 * Final structure:
 *   for tile = 0..N-2+32*(TSTEPS-1) step 32:
 *     for t = max(0, ⌊(tile-(N-2))/32⌋+1) .. min(TSTEPS, ⌊tile/32⌋+1):
 *       map i_ext: K1 body (extended by radius=1)
 *       map j:     K2 body
 */
struct Jacobi1DFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;
    structured_control_flow::Map* map_k1 = nullptr;
    structured_control_flow::Map* map_k2 = nullptr;
    structured_control_flow::For* loop_t = nullptr;

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

        // K1: map i = 0..N-2: B[1+i] = 0.333 * (A[i] + A[1+i] + A[2+i])
        auto& k1 = builder->add_map(
            time_loop.root(),
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k1 = &k1;
        {
            auto& block1 = builder->add_block(k1.root());
            auto& a_in1 = builder->add_access(block1, "A");
            auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp1_out = builder->add_access(block1, "tmp1");
            builder->add_computational_memlet(block1, a_in1, tasklet1, "_in1", {symbolic::symbol("i")}, desc_1d);
            builder->add_computational_memlet(
                block1, a_in1, tasklet1, "_in2", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, desc_1d
            );
            builder->add_computational_memlet(block1, tasklet1, "_out", tmp1_out, {});

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
            time_loop.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k2 = &k2;
        {
            auto& block1 = builder->add_block(k2.root());
            auto& b_in1 = builder->add_access(block1, "B");
            auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& tmp3_out = builder->add_access(block1, "tmp3");
            builder->add_computational_memlet(block1, b_in1, tasklet1, "_in1", {symbolic::symbol("j")}, desc_1d);
            builder->add_computational_memlet(
                block1, b_in1, tasklet1, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, desc_1d
            );
            builder->add_computational_memlet(block1, tasklet1, "_out", tmp3_out, {});

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

/// Multi-kernel diamond tiling pipeline:
///   Tile(K1,32) + Tile(K2,32) → cleanup → TileFusion → Skew(t,tile,32) → FM IC(t,tile)
TEST(DiamondTilingTest, Jacobi1D) {
    Jacobi1DFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Re-navigate after move
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* loop_t = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(loop_t, nullptr);
    ASSERT_EQ(loop_t->root().size(), 2);

    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // --- Step 1: Tile both Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k1, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k2, 32);
    am.invalidate_all();

    // --- Step 2: Cleanup ---
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After cleanup: for t: tile_k1 → tile_k2
    ASSERT_EQ(loop_t->root().size(), 2);
    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(tile_k1, nullptr);
    ASSERT_NE(tile_k2, nullptr);

    // --- Step 3: TileFusion ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_k1, *tile_k2);
    am.invalidate_all();

    // After fusion: for t: for tile_0: map K1_ext, map K2
    ASSERT_EQ(loop_t->root().size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&loop_t->root().at(0).first);
    ASSERT_NE(fused_tile, nullptr);
    ASSERT_EQ(fused_tile->root().size(), 2);

    // --- Step 4: LoopSkewing(t, tile, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *loop_t, *fused_tile, 32);
    am.invalidate_all();

    // --- Step 5: LoopInterchange(t, tile) with FM α=32 ---
    auto fused_tile_indvar_name = fused_tile->indvar()->get_name();
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_t, *fused_tile);
    am.invalidate_all();

    // After interchange: outer = tile, inner = t
    auto* outer = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->indvar()->get_name(), fused_tile_indvar_name);

    auto* inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->indvar()->get_name(), "t");

    // Inner t-loop should have two Maps as children (K1_ext and K2)
    ASSERT_EQ(inner->root().size(), 2);
    auto* k1_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(0).first);
    auto* k2_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(1).first);
    ASSERT_NE(k1_map, nullptr);
    ASSERT_NE(k2_map, nullptr);

    // Verify recorder history
    auto history = recorder.get_history();
    ASSERT_EQ(history.size(), 5);
    EXPECT_EQ(history[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[1]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[2]["transformation_type"], "TileFusion");
    EXPECT_EQ(history[2]["parameters"]["radius"], 1);
    EXPECT_EQ(history[3]["transformation_type"], "LoopSkewing");
    EXPECT_EQ(history[3]["parameters"]["skew_factor"], 32);
    EXPECT_EQ(history[4]["transformation_type"], "LoopInterchange");
}

/**
 * Builds a Jacobi-2D SDFG with two nested Maps inside a For time loop:
 *
 *   for t = 0..TSTEPS:
 *     map i = 1..N-1:
 *       map j = 1..N-1:
 *         B[i][j] = 0.2*(A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])
 *     map i = 1..N-1:
 *       map j = 1..N-1:
 *         A[i][j] = 0.2*(B[i][j] + B[i-1][j] + B[i+1][j] + B[i][j-1] + B[i][j+1])
 *
 * Diamond tiling pipeline (1D spatial - tile only i dimension):
 *   1. Tile(K1_i, 32) + Tile(K2_i, 32)
 *   2. Cleanup passes (DCE + SequenceFusion)
 *   3. TileFusion(K1_tile, K2_tile) — fuse tile loops, radius=1 in i-dimension
 *   4. LoopSkewing(t, tile, factor=32) — shift tile by 32*t
 *   5. LoopInterchange(t, tile) — tile loop becomes outermost
 *
 * Expected final structure:
 *   for tile_i = 0..N-2+32*(TSTEPS-1) step 32:
 *     for t = ...:
 *       map i_ext (extended by radius=1):
 *         map j: K1 body
 *       map i:
 *         map j: K2 body
 */
struct Jacobi2DFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;
    structured_control_flow::Map* map_k1 = nullptr;
    structured_control_flow::Map* map_k2 = nullptr;
    structured_control_flow::For* loop_t = nullptr;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("jacobi_2d", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("TSTEPS", sym_desc, true);
        builder->add_container("N", sym_desc, true);
        builder->add_container("t", sym_desc);
        builder->add_container("i_1", sym_desc);
        builder->add_container("j_1", sym_desc);
        builder->add_container("i_2", sym_desc);
        builder->add_container("j_2", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("k1_tmp1", elem_desc);
        builder->add_container("k1_tmp2", elem_desc);
        builder->add_container("k1_tmp3", elem_desc);
        builder->add_container("k1_tmp4", elem_desc);
        builder->add_container("k2_tmp1", elem_desc);
        builder->add_container("k2_tmp2", elem_desc);
        builder->add_container("k2_tmp3", elem_desc);
        builder->add_container("k2_tmp4", elem_desc);

        types::Array desc_1d(elem_desc, symbolic::symbol("N"));
        types::Pointer desc_2d(desc_1d);
        builder->add_container("A", desc_2d, true);
        builder->add_container("B", desc_2d, true);

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

        // K1: map i = 1..N-1: map j = 1..N-1: B[i][j] = 0.2*(5-point stencil of A)
        auto& k1 = builder->add_map(
            time_loop.root(),
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
            symbolic::integer(1),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k1 = &k1;
        {
            // Inner j map (data-parallel)
            auto& map_j = builder->add_map(
                k1.root(),
                symbolic::symbol("j_1"),
                symbolic::Lt(symbolic::symbol("j_1"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
                symbolic::integer(1),
                symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create()
            );
            {
                // Block 1: k1_tmp1 = A[i][j] + A[i-1][j]
                auto& block1 = builder->add_block(map_j.root());
                auto& a_in1 = builder->add_access(block1, "A");
                auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp1_out = builder->add_access(block1, "k1_tmp1");
                builder->add_computational_memlet(
                    block1, a_in1, tasklet1, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_1")}, desc_2d
                );
                builder->add_computational_memlet(
                    block1,
                    a_in1,
                    tasklet1,
                    "_in2",
                    {symbolic::sub(symbolic::symbol("i_1"), symbolic::integer(1)), symbolic::symbol("j_1")},
                    desc_2d
                );
                builder->add_computational_memlet(block1, tasklet1, "_out", tmp1_out, {});

                // Block 2: k1_tmp2 = k1_tmp1 + A[i+1][j]
                auto& block2 = builder->add_block(map_j.root());
                auto& tmp1_in = builder->add_access(block2, "k1_tmp1");
                auto& a_in2 = builder->add_access(block2, "A");
                auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp2_out = builder->add_access(block2, "k1_tmp2");
                builder->add_computational_memlet(block2, tmp1_in, tasklet2, "_in1", {});
                builder->add_computational_memlet(
                    block2,
                    a_in2,
                    tasklet2,
                    "_in2",
                    {symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1)), symbolic::symbol("j_1")},
                    desc_2d
                );
                builder->add_computational_memlet(block2, tasklet2, "_out", tmp2_out, {});

                // Block 3: k1_tmp3 = k1_tmp2 + A[i][j-1]
                auto& block3 = builder->add_block(map_j.root());
                auto& tmp2_in = builder->add_access(block3, "k1_tmp2");
                auto& a_in3 = builder->add_access(block3, "A");
                auto& tasklet3 = builder->add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp3_out = builder->add_access(block3, "k1_tmp3");
                builder->add_computational_memlet(block3, tmp2_in, tasklet3, "_in1", {});
                builder->add_computational_memlet(
                    block3,
                    a_in3,
                    tasklet3,
                    "_in2",
                    {symbolic::symbol("i_1"), symbolic::sub(symbolic::symbol("j_1"), symbolic::integer(1))},
                    desc_2d
                );
                builder->add_computational_memlet(block3, tasklet3, "_out", tmp3_out, {});

                // Block 4: k1_tmp4 = k1_tmp3 + A[i][j+1]
                auto& block4 = builder->add_block(map_j.root());
                auto& tmp3_in = builder->add_access(block4, "k1_tmp3");
                auto& a_in4 = builder->add_access(block4, "A");
                auto& tasklet4 = builder->add_tasklet(block4, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp4_out = builder->add_access(block4, "k1_tmp4");
                builder->add_computational_memlet(block4, tmp3_in, tasklet4, "_in1", {});
                builder->add_computational_memlet(
                    block4,
                    a_in4,
                    tasklet4,
                    "_in2",
                    {symbolic::symbol("i_1"), symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))},
                    desc_2d
                );
                builder->add_computational_memlet(block4, tasklet4, "_out", tmp4_out, {});

                // Block 5: B[i][j] = 0.2 * k1_tmp4
                auto& block5 = builder->add_block(map_j.root());
                auto& tmp4_in = builder->add_access(block5, "k1_tmp4");
                auto& const_node = builder->add_constant(block5, "0.2", elem_desc);
                auto& tasklet5 = builder->add_tasklet(block5, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
                auto& b_out = builder->add_access(block5, "B");
                builder->add_computational_memlet(block5, const_node, tasklet5, "_in1", {});
                builder->add_computational_memlet(block5, tmp4_in, tasklet5, "_in2", {});
                builder->add_computational_memlet(
                    block5, tasklet5, "_out", b_out, {symbolic::symbol("i_1"), symbolic::symbol("j_1")}, desc_2d
                );
            }
        }

        // K2: map i = 1..N-1: map j = 1..N-1: A[i][j] = 0.2*(5-point stencil of B)
        auto& k2 = builder->add_map(
            time_loop.root(),
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
            symbolic::integer(1),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k2 = &k2;
        {
            // Inner j map (data-parallel)
            auto& map_j = builder->add_map(
                k2.root(),
                symbolic::symbol("j_2"),
                symbolic::Lt(symbolic::symbol("j_2"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
                symbolic::integer(1),
                symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create()
            );
            {
                // Block 1: k2_tmp1 = B[i][j] + B[i-1][j]
                auto& block1 = builder->add_block(map_j.root());
                auto& b_in1 = builder->add_access(block1, "B");
                auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp1_out = builder->add_access(block1, "k2_tmp1");
                builder->add_computational_memlet(
                    block1, b_in1, tasklet1, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_2")}, desc_2d
                );
                builder->add_computational_memlet(
                    block1,
                    b_in1,
                    tasklet1,
                    "_in2",
                    {symbolic::sub(symbolic::symbol("i_2"), symbolic::integer(1)), symbolic::symbol("j_2")},
                    desc_2d
                );
                builder->add_computational_memlet(block1, tasklet1, "_out", tmp1_out, {});

                // Block 2: k2_tmp2 = k2_tmp1 + B[i+1][j]
                auto& block2 = builder->add_block(map_j.root());
                auto& tmp1_in = builder->add_access(block2, "k2_tmp1");
                auto& b_in2 = builder->add_access(block2, "B");
                auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp2_out = builder->add_access(block2, "k2_tmp2");
                builder->add_computational_memlet(block2, tmp1_in, tasklet2, "_in1", {});
                builder->add_computational_memlet(
                    block2,
                    b_in2,
                    tasklet2,
                    "_in2",
                    {symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1)), symbolic::symbol("j_2")},
                    desc_2d
                );
                builder->add_computational_memlet(block2, tasklet2, "_out", tmp2_out, {});

                // Block 3: k2_tmp3 = k2_tmp2 + B[i][j-1]
                auto& block3 = builder->add_block(map_j.root());
                auto& tmp2_in = builder->add_access(block3, "k2_tmp2");
                auto& b_in3 = builder->add_access(block3, "B");
                auto& tasklet3 = builder->add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp3_out = builder->add_access(block3, "k2_tmp3");
                builder->add_computational_memlet(block3, tmp2_in, tasklet3, "_in1", {});
                builder->add_computational_memlet(
                    block3,
                    b_in3,
                    tasklet3,
                    "_in2",
                    {symbolic::symbol("i_2"), symbolic::sub(symbolic::symbol("j_2"), symbolic::integer(1))},
                    desc_2d
                );
                builder->add_computational_memlet(block3, tasklet3, "_out", tmp3_out, {});

                // Block 4: k2_tmp4 = k2_tmp3 + B[i][j+1]
                auto& block4 = builder->add_block(map_j.root());
                auto& tmp3_in = builder->add_access(block4, "k2_tmp3");
                auto& b_in4 = builder->add_access(block4, "B");
                auto& tasklet4 = builder->add_tasklet(block4, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& tmp4_out = builder->add_access(block4, "k2_tmp4");
                builder->add_computational_memlet(block4, tmp3_in, tasklet4, "_in1", {});
                builder->add_computational_memlet(
                    block4,
                    b_in4,
                    tasklet4,
                    "_in2",
                    {symbolic::symbol("i_2"), symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))},
                    desc_2d
                );
                builder->add_computational_memlet(block4, tasklet4, "_out", tmp4_out, {});

                // Block 5: A[i][j] = 0.2 * k2_tmp4
                auto& block5 = builder->add_block(map_j.root());
                auto& tmp4_in = builder->add_access(block5, "k2_tmp4");
                auto& const_node = builder->add_constant(block5, "0.2", elem_desc);
                auto& tasklet5 = builder->add_tasklet(block5, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
                auto& a_out = builder->add_access(block5, "A");
                builder->add_computational_memlet(block5, const_node, tasklet5, "_in1", {});
                builder->add_computational_memlet(block5, tmp4_in, tasklet5, "_in2", {});
                builder->add_computational_memlet(
                    block5, tasklet5, "_out", a_out, {symbolic::symbol("i_2"), symbolic::symbol("j_2")}, desc_2d
                );
            }
        }
    }
};

/// Jacobi-2D with 1D spatial diamond tiling (tile only i dimension):
///   Tile(K1_i,32) + Tile(K2_i,32) → cleanup → TileFusion → Skew(t,tile,32) → IC(t,tile)
TEST(DiamondTilingTest, Jacobi2D_1DSpatial) {
    Jacobi2DFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Verify initial structure: for t { map i_1 { map j_1 { ... } }, map i_2 { map j_2 { ... } } }
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* loop_t = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(loop_t, nullptr);
    ASSERT_EQ(loop_t->root().size(), 2);

    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // Verify K1 has inner j map
    ASSERT_EQ(k1->root().size(), 1);
    auto* j1_map = dynamic_cast<structured_control_flow::Map*>(&k1->root().at(0).first);
    ASSERT_NE(j1_map, nullptr);

    // --- Step 1: Tile both Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k1, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k2, 32);
    am.invalidate_all();

    // --- Step 2: Cleanup ---
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After cleanup: for t { tile_k1 { ... }, tile_k2 { ... } }
    ASSERT_EQ(loop_t->root().size(), 2);
    auto* tile_k1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* tile_k2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(tile_k1, nullptr);
    ASSERT_NE(tile_k2, nullptr);

    // --- Step 3: TileFusion ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_k1, *tile_k2);
    am.invalidate_all();

    // After fusion: for t { for tile_0 { map K1_ext, map K2 } }
    ASSERT_EQ(loop_t->root().size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&loop_t->root().at(0).first);
    ASSERT_NE(fused_tile, nullptr);
    ASSERT_EQ(fused_tile->root().size(), 2);

    // --- Step 4: LoopSkewing(t, tile, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *loop_t, *fused_tile, 32);
    am.invalidate_all();

    // --- Step 5: LoopInterchange(t, tile) ---
    auto fused_tile_indvar_name = fused_tile->indvar()->get_name();
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_t, *fused_tile);
    am.invalidate_all();

    // After interchange: outer = tile, inner = t
    auto* outer = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->indvar()->get_name(), fused_tile_indvar_name);

    auto* inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->indvar()->get_name(), "t");

    // Inner t-loop should have two Maps as children (K1_ext and K2)
    ASSERT_EQ(inner->root().size(), 2);
    auto* k1_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(0).first);
    auto* k2_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(1).first);
    ASSERT_NE(k1_map, nullptr);
    ASSERT_NE(k2_map, nullptr);

    // Verify recorder history
    auto history = recorder.get_history();
    ASSERT_EQ(history.size(), 5);
    EXPECT_EQ(history[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[1]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[2]["transformation_type"], "TileFusion");
    EXPECT_EQ(history[2]["parameters"]["radius"], 1); // radius=1 for i-dimension stencil
    EXPECT_EQ(history[3]["transformation_type"], "LoopSkewing");
    EXPECT_EQ(history[3]["parameters"]["skew_factor"], 32);
    EXPECT_EQ(history[4]["transformation_type"], "LoopInterchange");
}

/**
 * Jacobi-2D with 2D spatial diamond tiling (tile both i and j dimensions):
 *
 * Phase 1 (i-dimension diamond tiling):
 *   1. Tile(i_1, 32) + Tile(i_2, 32)
 *   2. Cleanup
 *   3. TileFusion(tile_i_1, tile_i_2)
 *   4. LoopSkewing(t, tile_i, 32)
 *   5. LoopInterchange(t, tile_i)
 *
 * Phase 2 (j-dimension diamond tiling):
 *   6. Tile(j_1, 32) + Tile(j_2, 32)
 *   7. LoopInterchange(i_1, tile_j_1)
 *   8. LoopInterchange(i_2, tile_j_2)
 *   9. TileFusion(tile_j_1, tile_j_2)  -- now siblings after interchange
 *   10. LoopSkewing(t, tile_j, 32)
 *   11. LoopInterchange(t, tile_j)
 *
 * Expected final structure:
 *   for tile_i:
 *     for tile_j:
 *       for t:
 *         map i_1 { map j_1 { K1 } }
 *         map i_2 { map j_2 { K2 } }
 */
TEST(DiamondTilingTest, Jacobi2D_2DSpatial) {
    Jacobi2DFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Initial structure: for t { map i_1 { map j_1 }, map i_2 { map j_2 } }
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* loop_t = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(loop_t, nullptr);
    ASSERT_EQ(loop_t->root().size(), 2);

    auto* k1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* k2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // ========== PHASE 1: i-dimension diamond tiling ==========

    // --- Step 1: Tile both i-Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k1, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k2, 32);
    am.invalidate_all();

    // --- Step 2: Cleanup ---
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After cleanup: for t { tile_i_1 { i_1 { j_1 } }, tile_i_2 { i_2 { j_2 } } }
    ASSERT_EQ(loop_t->root().size(), 2);
    auto* tile_i_1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* tile_i_2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(tile_i_1, nullptr);
    ASSERT_NE(tile_i_2, nullptr);

    // --- Step 3: TileFusion on i-tiles ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_i_1, *tile_i_2);
    am.invalidate_all();

    // After fusion: for t { for fused_tile_i { map i_1{j_1}, map i_2{j_2} } }
    ASSERT_EQ(loop_t->root().size(), 1);
    auto* fused_tile_i = dynamic_cast<structured_control_flow::For*>(&loop_t->root().at(0).first);
    ASSERT_NE(fused_tile_i, nullptr);
    ASSERT_EQ(fused_tile_i->root().size(), 2);

    // --- Step 4: LoopSkewing(t, tile_i, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *loop_t, *fused_tile_i, 32);
    am.invalidate_all();

    // --- Step 5: LoopInterchange(t, tile_i) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_t, *fused_tile_i);
    am.invalidate_all();

    // After interchange: tile_i { t { map i_1{j_1}, map i_2{j_2} } }
    auto* outer_tile_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(outer_tile_i, nullptr);

    auto* inner_t = dynamic_cast<structured_control_flow::For*>(&outer_tile_i->root().at(0).first);
    ASSERT_NE(inner_t, nullptr);
    EXPECT_EQ(inner_t->indvar()->get_name(), "t");

    // ========== PHASE 2: j-dimension diamond tiling ==========

    // Navigate to the inner maps
    ASSERT_EQ(inner_t->root().size(), 2);
    auto* map_i_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    auto* map_i_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(map_i_1, nullptr);
    ASSERT_NE(map_i_2, nullptr);

    // Find inner j maps
    ASSERT_EQ(map_i_1->root().size(), 1);
    auto* map_j_1 = dynamic_cast<structured_control_flow::Map*>(&map_i_1->root().at(0).first);
    ASSERT_NE(map_j_1, nullptr);
    ASSERT_EQ(map_i_2->root().size(), 1);
    auto* map_j_2 = dynamic_cast<structured_control_flow::Map*>(&map_i_2->root().at(0).first);
    ASSERT_NE(map_j_2, nullptr);

    // --- Step 6: Tile both j-Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *map_j_1, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *map_j_2, 32);
    am.invalidate_all();

    // Cleanup
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After tiling: map i_1 { tile_j_1 { j_1 } }, map i_2 { tile_j_2 { j_2 } }
    // Re-navigate after tiling
    map_i_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    map_i_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(map_i_1, nullptr);
    ASSERT_NE(map_i_2, nullptr);

    ASSERT_EQ(map_i_1->root().size(), 1);
    auto* tile_j_1 = dynamic_cast<structured_control_flow::Map*>(&map_i_1->root().at(0).first);
    ASSERT_NE(tile_j_1, nullptr);
    ASSERT_EQ(map_i_2->root().size(), 1);
    auto* tile_j_2 = dynamic_cast<structured_control_flow::Map*>(&map_i_2->root().at(0).first);
    ASSERT_NE(tile_j_2, nullptr);

    // --- Step 7: LoopInterchange(i_1, tile_j_1) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *map_i_1, *tile_j_1);
    am.invalidate_all();

    // --- Step 8: LoopInterchange(i_2, tile_j_2) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *map_i_2, *tile_j_2);
    am.invalidate_all();

    // After interchange: t { tile_j_1{i_1{j_1}}, tile_j_2{i_2{j_2}} }
    // Now tile_j_1 and tile_j_2 are siblings!
    ASSERT_EQ(inner_t->root().size(), 2);
    tile_j_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    tile_j_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(tile_j_1, nullptr);
    ASSERT_NE(tile_j_2, nullptr);

    // --- Step 9: TileFusion(tile_j_1, tile_j_2) ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_j_1, *tile_j_2);
    am.invalidate_all();

    // After fusion: t { for fused_tile_j { i_1{j_1}, i_2{j_2} } }
    ASSERT_EQ(inner_t->root().size(), 1);
    auto* fused_tile_j = dynamic_cast<structured_control_flow::For*>(&inner_t->root().at(0).first);
    ASSERT_NE(fused_tile_j, nullptr);

    // --- Step 10: LoopSkewing(t, tile_j, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *inner_t, *fused_tile_j, 32);
    am.invalidate_all();

    // --- Step 11: LoopInterchange(t, tile_j) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *inner_t, *fused_tile_j);
    am.invalidate_all();

    // Final structure: tile_i { tile_j { t { i_1{j_1}, i_2{j_2} } } }
    auto* final_tile_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(final_tile_i, nullptr);

    ASSERT_EQ(final_tile_i->root().size(), 1);
    auto* final_tile_j = dynamic_cast<structured_control_flow::For*>(&final_tile_i->root().at(0).first);
    ASSERT_NE(final_tile_j, nullptr);

    ASSERT_EQ(final_tile_j->root().size(), 1);
    auto* final_t = dynamic_cast<structured_control_flow::For*>(&final_tile_j->root().at(0).first);
    ASSERT_NE(final_t, nullptr);
    EXPECT_EQ(final_t->indvar()->get_name(), "t");

    // Inner t should have two maps
    ASSERT_EQ(final_t->root().size(), 2);
}


/**
 * Builds a simplified FDTD-2D SDFG with two fusible kernels (K3 and K4) inside a For time loop:
 *
 *   for t = 0..TMAX:
 *     map i = 0..NX-1:
 *       map j = 1..NY-1:
 *         ex[i][j] = ex[i][j] - (hz[i][j] - hz[i][j-1])
 *     map i = 0..NX-2:
 *       map j = 0..NY-2:
 *         hz[i][j] = hz[i][j] + (ex[i][j+1] - ex[i][j])
 *
 * K3 writes ex, K4 reads ex — fusible pair with 1D j-dimension stencil radius=1.
 *
 * Diamond tiling pipeline (1D spatial - tile only i dimension):
 *   1. Tile(K3_i, 32) + Tile(K4_i, 32)
 *   2. Cleanup passes (DCE + SequenceFusion)
 *   3. TileFusion(K3_tile, K4_tile) — fuse into single tile loop, radius=1
 *   4. LoopSkewing(t, tile, factor=32) — shift tile by 32*t
 *   5. LoopInterchange(t, tile) — FM with α=32
 */
struct FDTD2DFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;
    structured_control_flow::Map* map_k3 = nullptr;
    structured_control_flow::Map* map_k4 = nullptr;
    structured_control_flow::For* loop_t = nullptr;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("fdtd_2d", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("TMAX", sym_desc, true);
        builder->add_container("NX", sym_desc, true);
        builder->add_container("NY", sym_desc, true);
        builder->add_container("t", sym_desc);
        builder->add_container("i_1", sym_desc);
        builder->add_container("j_1", sym_desc);
        builder->add_container("i_2", sym_desc);
        builder->add_container("j_2", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp1", elem_desc);
        builder->add_container("tmp2", elem_desc);

        types::Array desc_1d(elem_desc, symbolic::symbol("NY"));
        types::Pointer desc_2d(desc_1d);
        builder->add_container("ex", desc_2d, true);
        builder->add_container("hz", desc_2d, true);

        auto& root = builder->subject().root();

        // Time loop
        auto& time_loop = builder->add_for(
            root,
            symbolic::symbol("t"),
            symbolic::Lt(symbolic::symbol("t"), symbolic::symbol("TMAX")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("t"), symbolic::integer(1))
        );
        loop_t = &time_loop;

        // K3: map i = 0..NX-1: map j = 1..NY-1: ex[i][j] -= hz[i][j] - hz[i][j-1]
        auto& k3 = builder->add_map(
            time_loop.root(),
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("NX")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k3 = &k3;
        {
            auto& j_loop = builder->add_map(
                k3.root(),
                symbolic::symbol("j_1"),
                symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("NY")),
                symbolic::integer(1),
                symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create()
            );
            {
                // Block 1: tmp1 = hz[i][j] - hz[i][j-1]
                auto& block1 = builder->add_block(j_loop.root());
                auto& hz_in1 = builder->add_access(block1, "hz");
                auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
                auto& tmp1_out = builder->add_access(block1, "tmp1");
                builder->add_computational_memlet(
                    block1, hz_in1, tasklet1, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
                );
                builder->add_computational_memlet(
                    block1,
                    hz_in1,
                    tasklet1,
                    "_in2",
                    {symbolic::symbol("i_1"), symbolic::sub(symbolic::symbol("j_1"), symbolic::integer(1))}
                );
                builder->add_computational_memlet(block1, tasklet1, "_out", tmp1_out, {});

                // Block 2: ex[i][j] = ex[i][j] - tmp1
                auto& block2 = builder->add_block(j_loop.root());
                auto& ex_in = builder->add_access(block2, "ex");
                auto& tmp1_in = builder->add_access(block2, "tmp1");
                auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
                auto& ex_out = builder->add_access(block2, "ex");
                builder->add_computational_memlet(
                    block2, ex_in, tasklet2, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
                );
                builder->add_computational_memlet(block2, tmp1_in, tasklet2, "_in2", {});
                builder->add_computational_memlet(
                    block2, tasklet2, "_out", ex_out, {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
                );
            }
        }

        // K4: map i = 0..NX-2: map j = 0..NY-2: hz[i][j] += (ex[i][j+1] - ex[i][j])
        auto& k4 = builder->add_map(
            time_loop.root(),
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::sub(symbolic::symbol("NX"), symbolic::integer(1))),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        map_k4 = &k4;
        {
            auto& j_loop = builder->add_map(
                k4.root(),
                symbolic::symbol("j_2"),
                symbolic::Lt(symbolic::symbol("j_2"), symbolic::sub(symbolic::symbol("NY"), symbolic::integer(1))),
                symbolic::integer(0),
                symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create()
            );
            {
                // Block 1: tmp2 = ex[i][j+1] - ex[i][j]
                auto& block1 = builder->add_block(j_loop.root());
                auto& ex_in1 = builder->add_access(block1, "ex");
                auto& tasklet1 = builder->add_tasklet(block1, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
                auto& tmp2_out = builder->add_access(block1, "tmp2");
                builder->add_computational_memlet(
                    block1,
                    ex_in1,
                    tasklet1,
                    "_in1",
                    {symbolic::symbol("i_2"), symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))}
                );
                builder->add_computational_memlet(
                    block1, ex_in1, tasklet1, "_in2", {symbolic::symbol("i_2"), symbolic::symbol("j_2")}
                );
                builder->add_computational_memlet(block1, tasklet1, "_out", tmp2_out, {});

                // Block 2: hz[i][j] = hz[i][j] + tmp2
                auto& block2 = builder->add_block(j_loop.root());
                auto& hz_in = builder->add_access(block2, "hz");
                auto& tmp2_in = builder->add_access(block2, "tmp2");
                auto& tasklet2 = builder->add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
                auto& hz_out = builder->add_access(block2, "hz");
                builder->add_computational_memlet(
                    block2, hz_in, tasklet2, "_in1", {symbolic::symbol("i_2"), symbolic::symbol("j_2")}
                );
                builder->add_computational_memlet(block2, tmp2_in, tasklet2, "_in2", {});
                builder->add_computational_memlet(
                    block2, tasklet2, "_out", hz_out, {symbolic::symbol("i_2"), symbolic::symbol("j_2")}
                );
            }
        }
    }
};

/// FDTD-2D with 1D spatial diamond tiling (tile only i dimension):
///   Tile(K3_i,32) + Tile(K4_i,32) → cleanup → TileFusion → Skew(t,tile,32) → IC(t,tile)
TEST(DiamondTilingTest, FDTD2D_1DSpatial) {
    FDTD2DFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Verify initial structure: for t { map i_1 { map j_1 { ... } }, map i_2 { map j_2 { ... } } }
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* loop_t = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(loop_t, nullptr);
    ASSERT_EQ(loop_t->root().size(), 2);

    auto* k3 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* k4 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k3, nullptr);
    ASSERT_NE(k4, nullptr);

    // Verify K3 has inner j map
    ASSERT_EQ(k3->root().size(), 1);
    auto* j3_map = dynamic_cast<structured_control_flow::Map*>(&k3->root().at(0).first);
    ASSERT_NE(j3_map, nullptr);

    // --- Step 1: Tile both Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k3, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k4, 32);
    am.invalidate_all();

    // --- Step 2: Cleanup ---
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After cleanup: for t { tile_k3 { ... }, tile_k4 { ... } }
    ASSERT_EQ(loop_t->root().size(), 2);
    auto* tile_k3 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* tile_k4 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(tile_k3, nullptr);
    ASSERT_NE(tile_k4, nullptr);

    // --- Step 3: TileFusion ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_k3, *tile_k4);
    am.invalidate_all();

    // After fusion: for t { for tile_0 { map K3_ext, map K4 } }
    ASSERT_EQ(loop_t->root().size(), 1);
    auto* fused_tile = dynamic_cast<structured_control_flow::For*>(&loop_t->root().at(0).first);
    ASSERT_NE(fused_tile, nullptr);
    ASSERT_EQ(fused_tile->root().size(), 2);

    // --- Step 4: LoopSkewing(t, tile, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *loop_t, *fused_tile, 32);
    am.invalidate_all();

    // --- Step 5: LoopInterchange(t, tile) ---
    auto fused_tile_indvar_name = fused_tile->indvar()->get_name();
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_t, *fused_tile);
    am.invalidate_all();

    // After interchange: outer = tile, inner = t
    auto* outer = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->indvar()->get_name(), fused_tile_indvar_name);

    auto* inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->indvar()->get_name(), "t");

    // Inner t-loop should have two Maps as children (K3_ext and K4)
    ASSERT_EQ(inner->root().size(), 2);
    auto* k3_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(0).first);
    auto* k4_map = dynamic_cast<structured_control_flow::Map*>(&inner->root().at(1).first);
    ASSERT_NE(k3_map, nullptr);
    ASSERT_NE(k4_map, nullptr);

    // Verify recorder history
    auto history = recorder.get_history();
    ASSERT_EQ(history.size(), 5);
    EXPECT_EQ(history[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[1]["transformation_type"], "LoopTiling");
    EXPECT_EQ(history[2]["transformation_type"], "TileFusion");
    EXPECT_EQ(history[2]["parameters"]["radius"], 0); // radius=0: tiling on i, stencil on j
    EXPECT_EQ(history[3]["transformation_type"], "LoopSkewing");
    EXPECT_EQ(history[3]["parameters"]["skew_factor"], 32);
    EXPECT_EQ(history[4]["transformation_type"], "LoopInterchange");
}

/**
 * FDTD-2D with 2D spatial diamond tiling (tile both i and j dimensions):
 *
 * Note: FDTD-2D has asymmetric j-bounds:
 *   K3: j = 1..NY-1, K4: j = 0..NY-2
 * We need LoopShift to normalize before TileFusion.
 *
 * Phase 0 (normalize j-loops):
 *   0a. LoopShift(j_1) to shift init from 1 to 0
 *
 * Phase 1 (i-dimension diamond tiling):
 *   1. Tile(i_1, 32) + Tile(i_2, 32)
 *   2. Cleanup
 *   3. TileFusion(tile_i_1, tile_i_2)
 *   4. LoopSkewing(t, tile_i, 32)
 *   5. LoopInterchange(t, tile_i)
 *
 * Phase 2 (j-dimension diamond tiling):
 *   6. Tile(j_1, 32) + Tile(j_2, 32)
 *   7. LoopInterchange(i_1, tile_j_1)
 *   8. LoopInterchange(i_2, tile_j_2)
 *   9. TileFusion(tile_j_1, tile_j_2)
 *   10. LoopSkewing(t, tile_j, 32)
 *   11. LoopInterchange(t, tile_j)
 *
 * Expected final structure:
 *   for tile_i:
 *     for tile_j:
 *       for t:
 *         map i_1 { map j_1 { K3 } }
 *         map i_2 { map j_2 { K4 } }
 */
TEST(DiamondTilingTest, FDTD2D_2DSpatial) {
    FDTD2DFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Initial structure: for t { map i_1 { map j_1 }, map i_2 { map j_2 } }
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    auto* loop_t = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_NE(loop_t, nullptr);
    ASSERT_EQ(loop_t->root().size(), 2);

    auto* k3 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* k4 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k3, nullptr);
    ASSERT_NE(k4, nullptr);

    // ========== PHASE 0: Normalize j-loops with LoopShift ==========
    // K3 has j_1 = 1..NY-1, K4 has j_2 = 0..NY-2
    // Shift j_1 to start at 0

    // Find inner j maps
    ASSERT_EQ(k3->root().size(), 1);
    auto* map_j_1_orig = dynamic_cast<structured_control_flow::Map*>(&k3->root().at(0).first);
    ASSERT_NE(map_j_1_orig, nullptr);

    // Apply LoopShift to j_1 (shift from init=1 to init=0)
    recorder.apply<transformations::LoopShift>(builder, am, false, *map_j_1_orig);
    am.invalidate_all();

    // Cleanup after LoopShift
    passes::SymbolPropagation symbol_propagation;
    passes::DeadDataElimination dead_data;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= symbol_propagation.run(builder, am);
        applies |= dead_data.run(builder, am);
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // Re-fetch k3 and k4 after Phase 0 cleanup
    k3 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    k4 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(k3, nullptr);
    ASSERT_NE(k4, nullptr);

    // ========== PHASE 1: i-dimension diamond tiling ==========

    // --- Step 1: Tile both i-Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k3, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *k4, 32);
    am.invalidate_all();

    // --- Step 2: Cleanup ---
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After cleanup: for t { tile_i_1 { i_1 { j_1 } }, tile_i_2 { i_2 { j_2 } } }
    ASSERT_EQ(loop_t->root().size(), 2);
    auto* tile_i_1 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(0).first);
    auto* tile_i_2 = dynamic_cast<structured_control_flow::Map*>(&loop_t->root().at(1).first);
    ASSERT_NE(tile_i_1, nullptr);
    ASSERT_NE(tile_i_2, nullptr);

    // --- Step 3: TileFusion on i-tiles ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_i_1, *tile_i_2);
    am.invalidate_all();

    // After fusion: for t { for fused_tile_i { map i_1{j_1}, map i_2{j_2} } }
    ASSERT_EQ(loop_t->root().size(), 1);
    auto* fused_tile_i = dynamic_cast<structured_control_flow::For*>(&loop_t->root().at(0).first);
    ASSERT_NE(fused_tile_i, nullptr);
    ASSERT_EQ(fused_tile_i->root().size(), 2);

    // --- Step 4: LoopSkewing(t, tile_i, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *loop_t, *fused_tile_i, 32);
    am.invalidate_all();

    // --- Step 5: LoopInterchange(t, tile_i) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_t, *fused_tile_i);
    am.invalidate_all();

    // After interchange: tile_i { t { map i_1{j_1}, map i_2{j_2} } }
    auto* outer_tile_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(outer_tile_i, nullptr);

    auto* inner_t = dynamic_cast<structured_control_flow::For*>(&outer_tile_i->root().at(0).first);
    ASSERT_NE(inner_t, nullptr);
    EXPECT_EQ(inner_t->indvar()->get_name(), "t");

    // ========== PHASE 2: j-dimension diamond tiling ==========

    // Navigate to the inner maps
    ASSERT_EQ(inner_t->root().size(), 2);
    auto* map_i_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    auto* map_i_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(map_i_1, nullptr);
    ASSERT_NE(map_i_2, nullptr);

    // Find inner j maps
    ASSERT_EQ(map_i_1->root().size(), 1);
    auto* map_j_1 = dynamic_cast<structured_control_flow::Map*>(&map_i_1->root().at(0).first);
    ASSERT_NE(map_j_1, nullptr);
    ASSERT_EQ(map_i_2->root().size(), 1);
    auto* map_j_2 = dynamic_cast<structured_control_flow::Map*>(&map_i_2->root().at(0).first);
    ASSERT_NE(map_j_2, nullptr);

    // --- Step 6: Tile both j-Maps with tile size 32 ---
    recorder.apply<transformations::LoopTiling>(builder, am, false, *map_j_1, 32);
    am.invalidate_all();
    recorder.apply<transformations::LoopTiling>(builder, am, false, *map_j_2, 32);
    am.invalidate_all();

    // Cleanup
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);

    // After tiling: map i_1 { tile_j_1 { j_1 } }, map i_2 { tile_j_2 { j_2 } }
    // Re-navigate after tiling
    map_i_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    map_i_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(map_i_1, nullptr);
    ASSERT_NE(map_i_2, nullptr);

    ASSERT_EQ(map_i_1->root().size(), 1);
    auto* tile_j_1 = dynamic_cast<structured_control_flow::Map*>(&map_i_1->root().at(0).first);
    ASSERT_NE(tile_j_1, nullptr);
    ASSERT_EQ(map_i_2->root().size(), 1);
    auto* tile_j_2 = dynamic_cast<structured_control_flow::Map*>(&map_i_2->root().at(0).first);
    ASSERT_NE(tile_j_2, nullptr);

    // --- Step 7: LoopInterchange(i_1, tile_j_1) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *map_i_1, *tile_j_1);
    am.invalidate_all();

    // --- Step 8: LoopInterchange(i_2, tile_j_2) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *map_i_2, *tile_j_2);
    am.invalidate_all();

    // After interchange: t { tile_j_1{i_1{j_1}}, tile_j_2{i_2{j_2}} }
    // Now tile_j_1 and tile_j_2 are siblings!
    ASSERT_EQ(inner_t->root().size(), 2);
    tile_j_1 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(0).first);
    tile_j_2 = dynamic_cast<structured_control_flow::Map*>(&inner_t->root().at(1).first);
    ASSERT_NE(tile_j_1, nullptr);
    ASSERT_NE(tile_j_2, nullptr);

    // --- Step 9: TileFusion(tile_j_1, tile_j_2) ---
    recorder.apply<transformations::TileFusion>(builder, am, false, *tile_j_1, *tile_j_2);
    am.invalidate_all();

    // After fusion: t { for fused_tile_j { i_1{j_1}, i_2{j_2} } }
    ASSERT_EQ(inner_t->root().size(), 1);
    auto* fused_tile_j = dynamic_cast<structured_control_flow::For*>(&inner_t->root().at(0).first);
    ASSERT_NE(fused_tile_j, nullptr);

    // --- Step 10: LoopSkewing(t, tile_j, factor=32) ---
    recorder.apply<transformations::LoopSkewing>(builder, am, false, *inner_t, *fused_tile_j, 32);
    am.invalidate_all();

    // --- Step 11: LoopInterchange(t, tile_j) ---
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *inner_t, *fused_tile_j);
    am.invalidate_all();

    // Final structure: tile_i { tile_j { t { i_1{j_1}, i_2{j_2} } } }
    auto* final_tile_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    ASSERT_NE(final_tile_i, nullptr);

    ASSERT_EQ(final_tile_i->root().size(), 1);
    auto* final_tile_j = dynamic_cast<structured_control_flow::For*>(&final_tile_i->root().at(0).first);
    ASSERT_NE(final_tile_j, nullptr);

    ASSERT_EQ(final_tile_j->root().size(), 1);
    auto* final_t = dynamic_cast<structured_control_flow::For*>(&final_tile_j->root().at(0).first);
    ASSERT_NE(final_t, nullptr);
    EXPECT_EQ(final_t->indvar()->get_name(), "t");

    // Inner t should have two maps
    ASSERT_EQ(final_t->root().size(), 2);
}
