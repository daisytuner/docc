#include <gtest/gtest.h>
#include <sdfg/transformations/recorder.h>

#include <fstream>
#include <memory>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
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

        types::Array a_row(elem_desc, symbolic::symbol("K"));
        types::Pointer desc_A(a_row);
        types::Array b_row(elem_desc, symbolic::symbol("N"));
        types::Pointer desc_B(b_row);
        types::Array c_row(elem_desc, symbolic::symbol("N"));
        types::Pointer desc_C(c_row);

        builder->add_container("A", desc_A, true);
        builder->add_container("B", desc_B, true);
        builder->add_container("C", desc_C, true);

        auto& root = builder->subject().root();

        auto& i_loop = builder->add_for(
            root,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
        );

        auto& j_loop = builder->add_for(
            i_loop.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        auto& k_loop = builder->add_for(
            j_loop.root(),
            symbolic::symbol("k"),
            symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("K")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
        );

        // tmp_mul = A[i][k] * B[k][j]
        {
            auto& block = builder->add_block(k_loop.root());
            auto& a_in = builder->add_access(block, "A");
            auto& b_in = builder->add_access(block, "B");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder->add_computational_memlet(
                block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k")}, desc_A
            );
            builder->add_computational_memlet(
                block, b_in, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")}, desc_B
            );
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // C[i][j] += tmp_mul
        {
            auto& block = builder->add_block(k_loop.root());
            auto& c_in = builder->add_access(block, "C");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& c_out = builder->add_access(block, "C");
            builder->add_computational_memlet(
                block, c_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_C
            );
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(
                block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_C
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
TEST(BLASBlockingTest, GEMM_Phase1_TileLoopsOutermost) {
    GEMMFixture fixture;
    fixture.build();

    auto structured_sdfg = fixture.builder->move();
    builder::StructuredSDFGBuilder builder(structured_sdfg);
    analysis::AnalysisManager am(builder.subject());
    transformations::Recorder recorder;

    // Helper to print current loop structure
    auto print_loop_order = [&](const std::string& label) {
        std::cout << label << ": ";
        std::vector<std::string> order;
        structured_control_flow::ControlFlowNode* current = &builder.subject().root().at(0).first;
        while (auto* loop = dynamic_cast<structured_control_flow::For*>(current)) {
            order.push_back(loop->indvar()->get_name());
            if (loop->root().size() > 0) {
                current = &loop->root().at(0).first;
            } else {
                break;
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            std::cout << order[i];
            if (i < order.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
    };

    print_loop_order("Initial");

    // Step 1: LoopInterchange(j, k) -> i-k-j
    auto* loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* loop_j = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    auto* loop_k = dynamic_cast<structured_control_flow::For*>(&loop_j->root().at(0).first);

    std::cout << "Step 1: LoopInterchange(j, k)" << std::endl;
    recorder.apply<transformations::LoopInterchange>(builder, am, false, *loop_j, *loop_k);
    am.invalidate_all();
    print_loop_order("After Step 1");

    // Step 2: LoopTiling(j, NC=256)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);
    loop_j = dynamic_cast<structured_control_flow::For*>(&loop_k->root().at(0).first);

    std::cout << "Step 2: LoopTiling(j, NC=256)" << std::endl;
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_j, 256);
    am.invalidate_all();
    cleanup(builder, am);
    print_loop_order("After Step 2");

    // Step 3: LoopTiling(k, KC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    loop_k = dynamic_cast<structured_control_flow::For*>(&loop_i->root().at(0).first);

    std::cout << "Step 3: LoopTiling(k, KC=64)" << std::endl;
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_k, 64);
    am.invalidate_all();
    cleanup(builder, am);
    print_loop_order("After Step 3");

    // Step 4: LoopTiling(i, MC=64)
    loop_i = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);

    std::cout << "Step 4: LoopTiling(i, MC=64)" << std::endl;
    recorder.apply<transformations::LoopTiling>(builder, am, false, *loop_i, 64);
    am.invalidate_all();
    cleanup(builder, am);
    print_loop_order("After Step 4");

    // Current order: i_tile -> i -> k_tile -> k -> j_tile -> j
    // Step 5: LoopInterchange(i, k_tile)
    auto* i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    auto* i_inner = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    auto* k_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    std::cout << "Step 5: LoopInterchange(i, k_tile)" << std::endl;
    transformations::LoopInterchange ic5(*i_inner, *k_tile);
    if (ic5.can_be_applied(builder, am)) {
        recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *k_tile);
        am.invalidate_all();
        print_loop_order("After Step 5");
    } else {
        std::cout << "  NOT applicable (dependency)" << std::endl;
    }

    // Step 6: LoopInterchange(k, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    auto* k_inner = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);
    auto* j_tile = dynamic_cast<structured_control_flow::For*>(&k_inner->root().at(0).first);

    if (k_inner && j_tile) {
        std::cout << "Step 6: LoopInterchange(k, j_tile)" << std::endl;
        transformations::LoopInterchange ic6(*k_inner, *j_tile);
        if (ic6.can_be_applied(builder, am)) {
            recorder.apply<transformations::LoopInterchange>(builder, am, false, *k_inner, *j_tile);
            am.invalidate_all();
            print_loop_order("After Step 6");
        } else {
            std::cout << "  NOT applicable (dependency)" << std::endl;
        }
    }

    // Step 7: LoopInterchange(i, j_tile)
    i_tile = dynamic_cast<structured_control_flow::For*>(&builder.subject().root().at(0).first);
    k_tile = dynamic_cast<structured_control_flow::For*>(&i_tile->root().at(0).first);
    i_inner = dynamic_cast<structured_control_flow::For*>(&k_tile->root().at(0).first);
    j_tile = dynamic_cast<structured_control_flow::For*>(&i_inner->root().at(0).first);

    if (i_inner && j_tile) {
        std::cout << "Step 7: LoopInterchange(i, j_tile)" << std::endl;
        transformations::LoopInterchange ic7(*i_inner, *j_tile);
        if (ic7.can_be_applied(builder, am)) {
            recorder.apply<transformations::LoopInterchange>(builder, am, false, *i_inner, *j_tile);
            am.invalidate_all();
            print_loop_order("After Step 7");
        } else {
            std::cout << "  NOT applicable (dependency)" << std::endl;
        }
    }

    // Verify final structure
    std::cout << "\n=== Final Loop Structure ===" << std::endl;
    print_loop_order("Final");

    // Collect loop names in order
    std::vector<std::string> final_order;
    structured_control_flow::ControlFlowNode* current = &builder.subject().root().at(0).first;
    while (auto* loop = dynamic_cast<structured_control_flow::For*>(current)) {
        final_order.push_back(loop->indvar()->get_name());
        if (loop->root().size() > 0) {
            current = &loop->root().at(0).first;
        } else {
            break;
        }
    }

    // Should have 6 loops
    ASSERT_EQ(final_order.size(), 6u);

    // Print transformation history
    auto history = recorder.get_history();
    std::cout << "\nTransformations applied: " << history.size() << std::endl;
    for (size_t i = 0; i < history.size(); i++) {
        std::cout << "  " << i << ": " << history[i]["transformation_type"] << std::endl;
    }
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

        types::Array y_desc(elem_desc, symbolic::symbol("M"));
        types::Array x_desc(elem_desc, symbolic::symbol("N"));
        types::Array a_row(elem_desc, symbolic::symbol("N"));
        types::Pointer a_desc(a_row);

        builder->add_container("y", y_desc, true);
        builder->add_container("x", x_desc, true);
        builder->add_container("A", a_desc, true);

        auto& root = builder->subject().root();

        auto& i_loop = builder->add_for(
            root,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
        );

        auto& j_loop = builder->add_for(
            i_loop.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        // tmp_mul = A[i][j] * x[j]
        {
            auto& block = builder->add_block(j_loop.root());
            auto& a_in = builder->add_access(block, "A");
            auto& x_in = builder->add_access(block, "x");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder->add_computational_memlet(
                block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, a_desc
            );
            builder->add_computational_memlet(block, x_in, tasklet, "_in2", {symbolic::symbol("j")}, x_desc);
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // y[i] += tmp_mul
        {
            auto& block = builder->add_block(j_loop.root());
            auto& y_in = builder->add_access(block, "y");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& y_out = builder->add_access(block, "y");
            builder->add_computational_memlet(block, y_in, tasklet, "_in1", {symbolic::symbol("i")}, y_desc);
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(block, tasklet, "_out", y_out, {symbolic::symbol("i")}, y_desc);
        }
    }
};

// ============================================================================
// SYRK: C[i][j] += A[i][k] * A[j][k] with j <= i (triangular)
// ============================================================================

struct SYRKFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("syrk", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("N", sym_desc, true);
        builder->add_container("K", sym_desc, true);
        builder->add_container("i", sym_desc);
        builder->add_container("j", sym_desc);
        builder->add_container("k", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp_mul", elem_desc);

        types::Array a_row(elem_desc, symbolic::symbol("K"));
        types::Pointer desc_A(a_row);
        types::Array c_row(elem_desc, symbolic::symbol("N"));
        types::Pointer desc_C(c_row);

        builder->add_container("A", desc_A, true);
        builder->add_container("C", desc_C, true);

        auto& root = builder->subject().root();

        auto& i_loop = builder->add_for(
            root,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
        );

        // TRIANGULAR: j <= i
        auto& j_loop = builder->add_for(
            i_loop.root(),
            symbolic::symbol("j"),
            symbolic::Le(symbolic::symbol("j"), symbolic::symbol("i")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        auto& k_loop = builder->add_for(
            j_loop.root(),
            symbolic::symbol("k"),
            symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("K")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
        );

        // tmp_mul = A[i][k] * A[j][k]
        {
            auto& block = builder->add_block(k_loop.root());
            auto& a_in1 = builder->add_access(block, "A");
            auto& a_in2 = builder->add_access(block, "A");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder->add_computational_memlet(
                block, a_in1, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k")}, desc_A
            );
            builder->add_computational_memlet(
                block, a_in2, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")}, desc_A
            );
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // C[i][j] += tmp_mul
        {
            auto& block = builder->add_block(k_loop.root());
            auto& c_in = builder->add_access(block, "C");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& c_out = builder->add_access(block, "C");
            builder->add_computational_memlet(
                block, c_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_C
            );
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(
                block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_C
            );
        }
    }
};

// ============================================================================
// TRMM: B[i][j] += A[i][k] * B[k][j] with k = i+1..M (triangular)
// ============================================================================

struct TRMMFixture {
    std::unique_ptr<builder::StructuredSDFGBuilder> builder;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("trmm", FunctionType_CPU);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder->add_container("M", sym_desc, true);
        builder->add_container("N", sym_desc, true);
        builder->add_container("i", sym_desc);
        builder->add_container("j", sym_desc);
        builder->add_container("k", sym_desc);

        types::Scalar elem_desc(types::PrimitiveType::Double);
        builder->add_container("tmp_mul", elem_desc);

        types::Array a_row(elem_desc, symbolic::symbol("M"));
        types::Pointer desc_A(a_row);
        types::Array b_row(elem_desc, symbolic::symbol("N"));
        types::Pointer desc_B(b_row);

        builder->add_container("A", desc_A, true);
        builder->add_container("B", desc_B, true);

        auto& root = builder->subject().root();

        auto& i_loop = builder->add_for(
            root,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
        );

        auto& j_loop = builder->add_for(
            i_loop.root(),
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        // TRIANGULAR: k = i+1..M
        auto& k_loop = builder->add_for(
            j_loop.root(),
            symbolic::symbol("k"),
            symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
        );

        // tmp_mul = A[i][k] * B[k][j]
        {
            auto& block = builder->add_block(k_loop.root());
            auto& a_in = builder->add_access(block, "A");
            auto& b_in = builder->add_access(block, "B");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
            auto& tmp_out = builder->add_access(block, "tmp_mul");
            builder->add_computational_memlet(
                block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("k")}, desc_A
            );
            builder->add_computational_memlet(
                block, b_in, tasklet, "_in2", {symbolic::symbol("k"), symbolic::symbol("j")}, desc_B
            );
            builder->add_computational_memlet(block, tasklet, "_out", tmp_out, {});
        }

        // B[i][j] += tmp_mul
        {
            auto& block = builder->add_block(k_loop.root());
            auto& b_in = builder->add_access(block, "B");
            auto& tmp_in = builder->add_access(block, "tmp_mul");
            auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
            auto& b_out = builder->add_access(block, "B");
            builder->add_computational_memlet(
                block, b_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_B
            );
            builder->add_computational_memlet(block, tmp_in, tasklet, "_in2", {});
            builder->add_computational_memlet(
                block, tasklet, "_out", b_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_B
            );
        }
    }
};
