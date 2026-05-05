#include <gtest/gtest.h>
#include <sdfg/transformations/recorder.h>

#include <memory>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/in_local_storage.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/out_local_storage.h"

using namespace sdfg;

namespace {

void cleanup(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am) {
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);
}

// Find the LAST For loop in a sequence (computation path, not copy loops)
structured_control_flow::For* find_last_for(structured_control_flow::Sequence& seq) {
    structured_control_flow::For* last = nullptr;
    for (size_t idx = 0; idx < seq.size(); idx++) {
        if (auto* f = dynamic_cast<structured_control_flow::For*>(&seq.at(idx).first)) {
            last = f;
        }
    }
    return last;
}

// Collect loop indvar names following computation path (LAST child at each level)
std::vector<std::string> get_loop_order(builder::StructuredSDFGBuilder& builder) {
    std::vector<std::string> order;
    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    while (loop) {
        order.push_back(loop->indvar()->get_name());
        loop = find_last_for(loop->root());
    }
    return order;
}

// Find an access node for a container within a loop body
data_flow::AccessNode& find_access_node(
    analysis::AnalysisManager& am, structured_control_flow::StructuredLoop& loop, const std::string& container
) {
    auto& users = am.get<analysis::Users>();
    analysis::UsersView view(users, loop.root());
    auto accesses = view.uses(container);
    assert(!accesses.empty());
    auto* node = dynamic_cast<data_flow::AccessNode*>(accesses.front()->element());
    assert(node);
    return *node;
}

} // namespace

// ============================================================================
// GEMM: C[i][j] += A[i][k] * B[k][j]
// ============================================================================

struct GEMMFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("gemm", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("M", sym_desc, true);
        builder->add_container("N", sym_desc, true);
        builder->add_container("K", sym_desc, true);
        builder->add_container("i", sym_desc);
        builder->add_container("j", sym_desc);
        builder->add_container("k", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp_mul", elem_desc);

        // Flat pointers with linearized accesses
        types::Pointer ptr_desc(elem_desc);
        builder->add_container("A", ptr_desc, true); // A[i*K+k]
        builder->add_container("B", ptr_desc, true); // B[k*N+j]
        builder->add_container("C", ptr_desc, true); // C[i*N+j]

        auto& root = builder->subject().root();

        auto i = symbolic::symbol("i");
        auto j = symbolic::symbol("j");
        auto k = symbolic::symbol("k");
        auto M = symbolic::symbol("M");
        auto N = symbolic::symbol("N");
        auto K = symbolic::symbol("K");

        auto& i_loop =
            builder->add_for(root, i, symbolic::Lt(i, M), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));

        auto& j_loop = builder->add_for(
            i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1))
        );

        auto& k_loop = builder->add_for(
            j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1))
        );

        // tmp_mul = A[i*K+k] * B[k*N+j]
        {
            auto& block = builder->add_block(k_loop.root());
            auto& a_in = builder->add_access(block, "A");
            auto& b_in = builder->add_access(block, "B");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder
                ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::add(symbolic::mul(i, K), k)}, ptr_desc);
            builder
                ->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::add(symbolic::mul(k, N), j)}, ptr_desc);
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // C[i*N+j] += tmp_mul
        {
            auto& block = builder->add_block(k_loop.root());
            auto& c_in = builder->add_access(block, "C");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& c_out = builder->add_access(block, "C");
            builder
                ->add_computational_memlet(block, c_in, tasklet, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(
                block, tasklet, "_out", c_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc
            );
        }
    }
};

// ============================================================================
// GEMM Phase 1: Proper 6-Loop Nest with Tile Loops Outermost
// ============================================================================

/**
 * GEMM BLIS Phase 1 - Loop Nest Restructuring:
 *
 * Original:   i → j → k
 *
 * Target:     i_tile → k_tile → j_tile → i → k → j
 *             (all tile loops outermost, point loops innermost)
 *
 * Transformation sequence:
 *   1. LoopInterchange(j, k)      : i → k → j
 *   2. LoopTiling(j, NC=256)      : i → k → j_tile → j
 *   3. LoopTiling(k, KC=64)       : i → k_tile → k → j_tile → j
 *   4. LoopTiling(i, MC=64)       : i_tile → i → k_tile → k → j_tile → j
 *   5. LoopInterchange(i, k_tile) : i_tile → k_tile → i → k → j_tile → j
 *   6. LoopInterchange(k, j_tile) : i_tile → k_tile → i → j_tile → k → j
 *   7. LoopInterchange(i, j_tile) : i_tile → k_tile → j_tile → i → k → j
 *
 * Final pseudocode:
 *   for ic = 0..M step MC:        // i_tile
 *       for pc = 0..K step KC:    // k_tile
 *           for jc = 0..N step NC:// j_tile
 *               for i = ic..ic+MC:    // point loops
 *                   for k = pc..pc+KC:
 *                       for j = jc..jc+NC:
 *                           C[i][j] += A[i][k] * B[k][j]
 */
TEST(BlockingTest, GEMM_Phase1_Tiling) {
    GEMMFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Initial: i → j → k
    auto initial_order = get_loop_order(builder);
    ASSERT_EQ(initial_order.size(), 3u);
    ASSERT_EQ(initial_order[0], "i");
    ASSERT_EQ(initial_order[1], "j");
    ASSERT_EQ(initial_order[2], "k");

    // Step 1: LoopInterchange(j, k) → i → k → j
    auto* loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* loop_j = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    auto* loop_k = dynamic_cast<structured_control_flow::For*>(&loop_j->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*loop_j, *loop_k).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_j, *loop_k);
    am.invalidate_all();

    // Step 2: LoopTiling(j, NC=256)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    loop_j = dynamic_cast<structured_control_flow::For*>(&loop_k->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_j, 256).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_j, 256);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 3: LoopTiling(k, KC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_k, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_k, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 4: LoopTiling(i, MC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_i, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_i, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 5: LoopInterchange(i, k_tile)
    auto* i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* i_inner = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    auto* k_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *k_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *k_tile);
    am.invalidate_all();

    // Step 6: LoopInterchange(k, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    auto* k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    auto* j_tile = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*k_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *k_inner, *j_tile);
    am.invalidate_all();

    // Step 7: LoopInterchange(i, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    j_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *j_tile);
    am.invalidate_all();

    // Verify final loop structure: i_tile → k_tile → j_tile → i → k → j
    auto final_order = get_loop_order(builder);
    ASSERT_EQ(final_order.size(), 6u);
    // Tile loops (outer 3): verify they contain "i", "k", "j" respectively
    // The names include original var name after tiling
    EXPECT_TRUE(final_order[0].find("i") != std::string::npos);
    EXPECT_TRUE(final_order[1].find("k") != std::string::npos);
    EXPECT_TRUE(final_order[2].find("j") != std::string::npos);
    // Point loops (inner 3)
    EXPECT_TRUE(final_order[3].find("i") != std::string::npos);
    EXPECT_TRUE(final_order[4].find("k") != std::string::npos);
    EXPECT_TRUE(final_order[5].find("j") != std::string::npos);

    // Should have 7 transformations total
    EXPECT_EQ(recorder.get_history().size(), 7u);
}

// ============================================================================
// GEMM Phase 2: Packing A and B with InLocalStorage
// ============================================================================

/**
 * GEMM BLIS Phase 2 - Packing:
 *
 * After Phase 1 loop restructuring:
 *   i_tile → k_tile → j_tile → i → k → j
 *
 * Phase 2 adds packing:
 *   8. InLocalStorage(k_tile, "A")  : Pack A[MC×KC] panel before j_tile loop
 *   9. InLocalStorage(j_tile, "B")  : Pack B[KC×NC] panel before point loops
 */
TEST(BlockingTest, GEMM_Phase2_Packing) {
    GEMMFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // ========================================================================
    // Phase 1: Loop restructuring (same as GEMM_Phase1_Tiling)
    // ========================================================================

    // Step 1: LoopInterchange(j, k) → i → k → j
    auto* loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* loop_j = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    auto* loop_k = dynamic_cast<structured_control_flow::For*>(&loop_j->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*loop_j, *loop_k).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_j, *loop_k);
    am.invalidate_all();

    // Step 2: LoopTiling(j, NC=256)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    loop_j = dynamic_cast<structured_control_flow::For*>(&loop_k->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_j, 256).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_j, 256);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 3: LoopTiling(k, KC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_k, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_k, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 4: LoopTiling(i, MC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_i, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_i, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 5: LoopInterchange(i, k_tile)
    auto* i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* i_inner = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    auto* k_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *k_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *k_tile);
    am.invalidate_all();

    // Step 6: LoopInterchange(k, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    auto* k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    auto* j_tile = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*k_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *k_inner, *j_tile);
    am.invalidate_all();

    // Step 7: LoopInterchange(i, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    j_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *j_tile);
    am.invalidate_all();

    // ========================================================================
    // Phase 2: Packing with InLocalStorage
    //
    // BLIS packing levels:
    //   A[i*K+k] doesn't depend on j → apply at j_tile level
    //     → tile extents MC×KC (integer), copy placed before j_tile = correct
    //   B[k*N+j] doesn't depend on i → apply at i point loop level
    //     → tile extents KC×NC (integer), copy placed before i_loop = correct
    // ========================================================================

    // Navigate: i_tile → k_tile → j_tile → i → k → j
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    structured_control_flow::For* j_tile_loop = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first
    );
    ASSERT_NE(j_tile_loop, nullptr);

    // Step 8: InLocalStorage(j_tile, "A") - Pack A[MC×KC] panel
    // A doesn't depend on j_tile's indvar, so union across j_tile iterations = MC×KC
    auto& a_access_p2 = find_access_node(am, *j_tile_loop, "A");
    ASSERT_TRUE(transformations::InLocalStorage(*j_tile_loop, a_access_p2).can_be_applied(builder, am));
    recorder.apply<transformations::InLocalStorage>(builder, am, false, *j_tile_loop, a_access_p2);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 9: InLocalStorage(i_loop, "B") - Pack B[KC×NC] panel
    // B doesn't depend on i's indvar, so union across i iterations = KC×NC
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    ASSERT_NE(j_tile_loop, nullptr);
    auto* i_point = find_last_for(j_tile_loop->root());
    ASSERT_NE(i_point, nullptr);

    auto& b_access_p2 = find_access_node(am, *i_point, "B");
    ASSERT_TRUE(transformations::InLocalStorage(*i_point, b_access_p2).can_be_applied(builder, am));
    recorder.apply<transformations::InLocalStorage>(builder, am, false, *i_point, b_access_p2);
    am.invalidate_all();
    cleanup(builder, am);

    // ========================================================================
    // Verification
    // ========================================================================

    // Verify containers were created
    EXPECT_TRUE(builder.subject().exists("__daisy_in_local_storage_A0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_in_local_storage_B0"));

    // Verify 9 transformations total
    EXPECT_EQ(recorder.get_history().size(), 9u);
}

// ============================================================================
// GEMM Phase 3: Register Blocking (Micro-Kernel)
// ============================================================================

/**
 * GEMM BLIS Phase 3 - Register Blocking:
 *
 * Transformation sequence (after Phase 2):
 *   10. LoopTiling(i, MR=6)        : ... → i_micro → i → k → j
 *   11. LoopTiling(j, NR=8)        : ... → i_micro → i → k → j_micro → j
 *   12. LoopInterchange(i, k)      : ... → i_micro → k → i → j_micro → j
 *   13. LoopInterchange(i, j_micro): ... → i_micro → k → j_micro → i → j
 *   14. LoopInterchange(k, j_micro): ... → i_micro → j_micro → k → i → j
 *   15. OutLocalStorage(i_micro, "C"): register tile for C[MR×NR]
 */
TEST(BlockingTest, GEMM_Phase3_RegisterBlocking) {
    GEMMFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // ========================================================================
    // Phase 1: Loop restructuring
    // ========================================================================

    // Step 1: LoopInterchange(j, k) → i → k → j
    auto* loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* loop_j = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    auto* loop_k = dynamic_cast<structured_control_flow::For*>(&loop_j->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*loop_j, *loop_k).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_j, *loop_k);
    am.invalidate_all();

    // Step 2: LoopTiling(j, NC=256)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    loop_j = dynamic_cast<structured_control_flow::For*>(&loop_k->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_j, 256).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_j, 256);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 3: LoopTiling(k, KC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_k, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_k, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 4: LoopTiling(i, MC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);

    ASSERT_TRUE(transformations::LoopTiling(*loop_i, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_i, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 5: LoopInterchange(i, k_tile)
    auto* i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* i_inner = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    auto* k_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *k_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *k_tile);
    am.invalidate_all();

    // Step 6: LoopInterchange(k, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    auto* k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    auto* j_tile = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*k_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *k_inner, *j_tile);
    am.invalidate_all();

    // Step 7: LoopInterchange(i, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    j_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *j_tile).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *j_tile);
    am.invalidate_all();

    // ========================================================================
    // Phase 2: Packing with InLocalStorage (same as Phase2 test)
    // ========================================================================

    // Navigate: i_tile → k_tile → j_tile → i → k → j
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    auto* j_tile_loop = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    ASSERT_NE(j_tile_loop, nullptr);

    // Step 8: InLocalStorage(j_tile, "A") - A doesn't depend on j
    auto& a_access_p3 = find_access_node(am, *j_tile_loop, "A");
    ASSERT_TRUE(transformations::InLocalStorage(*j_tile_loop, a_access_p3).can_be_applied(builder, am));
    recorder.apply<transformations::InLocalStorage>(builder, am, false, *j_tile_loop, a_access_p3);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 9: InLocalStorage(i_loop, "B") - B doesn't depend on i
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    ASSERT_NE(j_tile_loop, nullptr);
    auto* i_point = find_last_for(j_tile_loop->root());
    ASSERT_NE(i_point, nullptr);

    auto& b_access_p3 = find_access_node(am, *i_point, "B");
    ASSERT_TRUE(transformations::InLocalStorage(*i_point, b_access_p3).can_be_applied(builder, am));
    recorder.apply<transformations::InLocalStorage>(builder, am, false, *i_point, b_access_p3);
    am.invalidate_all();
    cleanup(builder, am);

    // ========================================================================
    // Phase 3: Register Blocking
    // ========================================================================

    // Find the i point loop inside j_tile (last For in j_tile's root)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    i_point = find_last_for(j_tile_loop->root());
    ASSERT_NE(i_point, nullptr);

    // Step 10: LoopTiling(i, MR=6)
    ASSERT_TRUE(transformations::LoopTiling(*i_point, 6).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *i_point, 6);
    am.invalidate_all();
    cleanup(builder, am);

    // Navigate to j loop: i_tile → k_tile → j_tile → i_micro → i → k → j
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    auto* i_micro = find_last_for(j_tile_loop->root());
    ASSERT_NE(i_micro, nullptr);

    i_inner = dynamic_cast<structured_control_flow::For*>(&i_micro->root().at(0).first);
    ASSERT_NE(i_inner, nullptr);

    k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    ASSERT_NE(k_inner, nullptr);

    auto* j_inner = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);
    ASSERT_NE(j_inner, nullptr);

    // Step 11: LoopTiling(j, NR=8)
    ASSERT_TRUE(transformations::LoopTiling(*j_inner, 8).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *j_inner, 8);
    am.invalidate_all();
    cleanup(builder, am);

    // Re-navigate: i_micro → i → k → j_micro → j
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    i_micro = find_last_for(j_tile_loop->root());
    i_inner = dynamic_cast<structured_control_flow::For*>(&i_micro->root().at(0).first);
    k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    auto* j_micro = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    // Step 12: LoopInterchange(i, k) → i_micro → k → i → j_micro → j
    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *k_inner).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *k_inner);
    am.invalidate_all();

    // Re-navigate after interchange
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    i_micro = find_last_for(j_tile_loop->root());
    k_inner = dynamic_cast<structured_control_flow::For*>(&i_micro->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);
    j_micro = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    // Step 13: LoopInterchange(i, j_micro) → i_micro → k → j_micro → i → j
    ASSERT_TRUE(transformations::LoopInterchange(*i_inner, *j_micro).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *j_micro);
    am.invalidate_all();

    // Re-navigate after interchange
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    i_micro = find_last_for(j_tile_loop->root());
    k_inner = dynamic_cast<structured_control_flow::For*>(&i_micro->root().at(0).first);
    j_micro = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    // Step 14: LoopInterchange(k, j_micro) → i_micro → j_micro → k → i → j
    ASSERT_TRUE(transformations::LoopInterchange(*k_inner, *j_micro).can_be_applied(builder, am));
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *k_inner, *j_micro);
    am.invalidate_all();

    // Step 15: OutLocalStorage(k_loop, "C") - register tile MR×NR
    // C[i*N+j] doesn't depend on k → union across k iterations = MR×NR
    // After interchanges: i_micro → j_micro → k → i → j
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    j_tile_loop = find_last_for(k_tile->root());
    i_micro = find_last_for(j_tile_loop->root());
    j_micro = dynamic_cast<structured_control_flow::For*>(&i_micro->root().at(0).first);
    auto* k_point = dynamic_cast<structured_control_flow::For*>(&j_micro->root().at(0).first);
    ASSERT_NE(k_point, nullptr);

    auto& c_access_p3 = find_access_node(am, *k_point, "C");
    ASSERT_TRUE(transformations::OutLocalStorage(*k_point, c_access_p3).can_be_applied(builder, am));
    recorder.apply<transformations::OutLocalStorage>(builder, am, false, *k_point, c_access_p3);
    am.invalidate_all();
    cleanup(builder, am);

    // ========================================================================
    // Verification
    // ========================================================================

    // Verify containers were created
    EXPECT_TRUE(builder.subject().exists("__daisy_in_local_storage_A0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_in_local_storage_B0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));

    // Verify loop structure: computation path should have at least 5 loops
    // (OutLocalStorage adds init/writeback loops outside the main path)
    auto final_order = get_loop_order(builder);
    EXPECT_GE(final_order.size(), 5u);

    // Verify 15 transformations total
    EXPECT_EQ(recorder.get_history().size(), 15u);
}

// ============================================================================
// GEMV: y[i] += A[i][j] * x[j]
// ============================================================================

struct GEMVFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("gemv", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("M", sym_desc, true);
        builder->add_container("N", sym_desc, true);
        builder->add_container("i", sym_desc);
        builder->add_container("j", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp_mul", elem_desc);

        // Flat pointers with linearized accesses
        types::Pointer ptr_desc(elem_desc);
        builder->add_container("y", ptr_desc, true); // y[i]
        builder->add_container("x", ptr_desc, true); // x[j]
        builder->add_container("A", ptr_desc, true); // A[i*N+j]

        auto& root = builder->subject().root();

        auto i = symbolic::symbol("i");
        auto j = symbolic::symbol("j");
        auto M = symbolic::symbol("M");
        auto N = symbolic::symbol("N");

        auto& i_loop =
            builder->add_for(root, i, symbolic::Lt(i, M), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));

        auto& j_loop = builder->add_for(
            i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1))
        );

        // tmp_mul = A[i*N+j] * x[j]
        {
            auto& block = builder->add_block(j_loop.root());
            auto& a_in = builder->add_access(block, "A");
            auto& x_in = builder->add_access(block, "x");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder
                ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
            builder->add_computational_memlet(block, x_in, tasklet, "_in2", {j}, ptr_desc);
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // y[i] += tmp_mul
        {
            auto& block = builder->add_block(j_loop.root());
            auto& y_in = builder->add_access(block, "y");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& y_out = builder->add_access(block, "y");
            builder->add_computational_memlet(block, y_in, tasklet, "_in1", {i}, ptr_desc);
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(block, tasklet, "_out", y_out, {i}, ptr_desc);
        }
    }
};

// ============================================================================
// GEMV Optimized: y[i] += A[i*N+j] * x[j]
// ============================================================================

/**
 * GEMV Optimization Pipeline:
 *
 * Transformation sequence:
 *   1. OutLocalStorage(j_loop, "y")  - scalar accumulator for y[i]
 *   2. LoopTiling(j, JC=64)          - tile j for cache blocking
 *   3. InLocalStorage(j_inner, "x")  - pack x[JC] per tile
 *      (x[j] at j_inner level has extent JC, the tile size)
 */
TEST(BlockingTest, GEMV_Optimized) {
    GEMVFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Initial: i → j
    auto initial_order = get_loop_order(builder);
    ASSERT_EQ(initial_order.size(), 2u);
    ASSERT_EQ(initial_order[0], "i");
    ASSERT_EQ(initial_order[1], "j");

    // Get loops
    auto* i_loop = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* j_loop = dynamic_cast<structured_control_flow::For*>(&i_loop->root().at(0).first);

    // Step 1: OutLocalStorage(j_loop, "y") - scalar accumulator for y[i]
    auto& y_access = find_access_node(am, *j_loop, "y");
    ASSERT_TRUE(transformations::OutLocalStorage(*j_loop, y_access).can_be_applied(builder, am));
    recorder.apply<transformations::OutLocalStorage>(builder, am, false, *j_loop, y_access);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 2: LoopTiling(j, JC=64) - tile j for cache blocking
    i_loop = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    j_loop = find_last_for(i_loop->root());
    ASSERT_NE(j_loop, nullptr);

    ASSERT_TRUE(transformations::LoopTiling(*j_loop, 64).can_be_applied(builder, am));
    recorder.apply<transformations::LoopTiling>(builder, am, false, *j_loop, 64);
    am.invalidate_all();
    cleanup(builder, am);

    // Step 3: InLocalStorage(j_inner, "x") - pack x[JC] per tile
    // x[j] at j_inner level: j ranges j_tile..min(j_tile+JC, N) → extent JC
    i_loop = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* j_tile = find_last_for(i_loop->root());
    ASSERT_NE(j_tile, nullptr);
    auto* j_inner = find_last_for(j_tile->root());
    ASSERT_NE(j_inner, nullptr);

    auto& x_access = find_access_node(am, *j_inner, "x");
    ASSERT_TRUE(transformations::InLocalStorage(*j_inner, x_access).can_be_applied(builder, am));
    recorder.apply<transformations::InLocalStorage>(builder, am, false, *j_inner, x_access);
    am.invalidate_all();
    cleanup(builder, am);

    // ========================================================================
    // Verification
    // ========================================================================

    // Verify containers were created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_y0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_in_local_storage_x0"));

    // Verify loop structure: i → j_tile → j (3 loops)
    auto final_order = get_loop_order(builder);
    EXPECT_EQ(final_order.size(), 3u);

    // Verify 3 transformations total
    EXPECT_EQ(recorder.get_history().size(), 3u);
}
