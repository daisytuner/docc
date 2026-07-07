#include "sdfg/analysis/loop_analysis.h"

#include <gtest/gtest.h>

#include "loop_info_debug_dump.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"

using namespace sdfg;

TEST(LoopAnalysisTest, Children_nested3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto children_i = loop_analysis.children(&loop_i);
    EXPECT_EQ(children_i.size(), 1);
    EXPECT_EQ(children_i.at(0), &loop_j);

    auto children_j = loop_analysis.children(&loop_j);
    EXPECT_EQ(children_j.size(), 1);
    EXPECT_EQ(children_j.at(0), &loop_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, Children_nested2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto children_i = loop_analysis.children(&loop_i);
    EXPECT_EQ(children_i.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto node : children_i) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, LoopTreePath_single) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.loop_tree_paths(&loop_i);
    EXPECT_EQ(path.size(), 1);
    EXPECT_EQ(path.back().size(), 3);
    EXPECT_EQ(path.back().at(0), &loop_i);
    EXPECT_EQ(path.back().at(1), &loop_j);
    EXPECT_EQ(path.back().at(2), &loop_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, LoopTreePath_split) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.loop_tree_paths(&loop_i);
    EXPECT_EQ(path.size(), 2);
    EXPECT_EQ(path.front().size(), 2);
    EXPECT_EQ(path.front().at(0), &loop_i);

    EXPECT_EQ(path.back().size(), 2);
    EXPECT_EQ(path.back().at(0), &loop_i);

    EXPECT_TRUE(
        path.front().at(1) == &loop_j && path.back().at(1) == &loop_k ||
        path.front().at(1) == &loop_k && path.back().at(1) == &loop_j
    );
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, descendants_nested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_j.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.descendants(&loop_i);
    EXPECT_EQ(path.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto& node : path) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, descendants_concatenated) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_k = builder.add_for(loop_i.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto path = loop_analysis.descendants(&loop_i);
    EXPECT_EQ(path.size(), 2);
    bool found_j = false;
    bool found_k = false;
    for (auto node : path) {
        if (node == &loop_j) {
            found_j = true;
        } else if (node == &loop_k) {
            found_k = true;
        }
    }
    EXPECT_TRUE(found_j);
    EXPECT_TRUE(found_k);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, outermost_loops) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    builder.add_container("i_1", desc_symbols);
    builder.add_container("i_2", desc_symbols);
    builder.add_container("i_3", desc_symbols);
    builder.add_container("i_4", desc_symbols);
    builder.add_container("j_1", desc_symbols);
    builder.add_container("j_2", desc_symbols);
    builder.add_container("j_3", desc_symbols);
    builder.add_container("j_4", desc_symbols);
    builder.add_container("N", desc_symbols, true);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);
    builder.add_container("A", desc_2d, true);
    builder.add_container("u1", desc_1d, true);
    builder.add_container("u2", desc_1d, true);
    builder.add_container("v1", desc_1d, true);
    builder.add_container("v2", desc_1d, true);
    builder.add_container("x", desc_1d, true);
    builder.add_container("y", desc_1d, true);
    builder.add_container("z", desc_1d, true);
    builder.add_container("w", desc_1d, true);

    auto& root = builder.subject().root();

    {
        auto& loop_i_1 = builder.add_for(
            root,
            symbolic::symbol("i_1"),
            symbolic::Lt(symbolic::symbol("i_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_1"), symbolic::integer(1))
        );
        auto& body_i_1 = loop_i_1.root();
        auto& loop_j_1 = builder.add_for(
            body_i_1,
            symbolic::symbol("j_1"),
            symbolic::Lt(symbolic::symbol("j_1"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
        );
        auto& body_j_1 = loop_j_1.root();

        builder.add_container("tmp_1", desc_element);

        auto& block = builder.add_block(body_j_1);
        auto& u1_node = builder.add_access(block, "u1");
        auto& v1_node = builder.add_access(block, "v1");
        auto& tmp_node = builder.add_access(block, "tmp_1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, u1_node, tasklet, "_in1", {symbolic::symbol("i_1")});
        builder.add_computational_memlet(block, v1_node, tasklet, "_in2", {symbolic::symbol("j_1")});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_node, {});

        builder.add_container("tmp_2", desc_element);

        auto& block2 = builder.add_block(body_j_1);
        auto& u2_node = builder.add_access(block2, "u2");
        auto& v2_node = builder.add_access(block2, "v2");
        auto& tmp2_node = builder.add_access(block2, "tmp_2");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block2, u2_node, tasklet2, "_in1", {symbolic::symbol("i_1")});
        builder.add_computational_memlet(block2, v2_node, tasklet2, "_in2", {symbolic::symbol("j_1")});
        builder.add_computational_memlet(block2, tasklet2, "_out", tmp2_node, {});

        builder.add_container("tmp_3", desc_element);

        auto& block3 = builder.add_block(body_j_1);
        auto& tmp_node_1 = builder.add_access(block3, "tmp_1");
        auto& tmp2_node_1 = builder.add_access(block3, "tmp_2");
        auto& tmp3_node = builder.add_access(block3, "tmp_3");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block3, tmp_node_1, tasklet3, "_in1", {});
        builder.add_computational_memlet(block3, tmp2_node_1, tasklet3, "_in2", {});
        builder.add_computational_memlet(block3, tasklet3, "_out", tmp3_node, {});

        auto& A_node = builder.add_access(block3, "A");
        auto& A_node_out = builder.add_access(block3, "A");
        auto& tasklet4 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(
            block3, A_node, tasklet4, "_in1", {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
        );
        builder.add_computational_memlet(block3, tmp3_node, tasklet4, "_in2", {});
        builder.add_computational_memlet(
            block3, tasklet4, "_out", A_node_out, {symbolic::symbol("i_1"), symbolic::symbol("j_1")}
        );
    }

    {
        auto& loop_i_2 = builder.add_for(
            root,
            symbolic::symbol("i_2"),
            symbolic::Lt(symbolic::symbol("i_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_2"), symbolic::integer(1))
        );
        auto& body_i_2 = loop_i_2.root();
        auto& loop_j_2 = builder.add_for(
            body_i_2,
            symbolic::symbol("j_2"),
            symbolic::Lt(symbolic::symbol("j_2"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
        );
        auto& body_j_2 = loop_j_2.root();

        auto& block = builder.add_block(body_j_2);
        auto& x_node_in = builder.add_access(block, "x");
        auto& x_node_out = builder.add_access(block, "x");
        auto& y_node = builder.add_access(block, "y");
        auto& A_node = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, x_node_in, tasklet, "_in3", {symbolic::symbol("i_2")});
        builder
            .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("i_2")});
        builder.add_computational_memlet(block, y_node, tasklet, "_in2", {symbolic::symbol("j_2")});
        builder.add_computational_memlet(block, tasklet, "_out", x_node_out, {symbolic::symbol("i_2")});
    }

    {
        auto& loop_i_3 = builder.add_for(
            root,
            symbolic::symbol("i_3"),
            symbolic::Lt(symbolic::symbol("i_3"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_3"), symbolic::integer(1))
        );
        auto& body_i_3 = loop_i_3.root();

        auto& block = builder.add_block(body_i_3);
        auto& x_node_in = builder.add_access(block, "x");
        auto& x_node_out = builder.add_access(block, "x");
        auto& z_node = builder.add_access(block, "z");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, x_node_in, tasklet, "_in1", {symbolic::symbol("i_3")});
        builder.add_computational_memlet(block, z_node, tasklet, "_in2", {symbolic::symbol("i_3")});
        builder.add_computational_memlet(block, tasklet, "_out", x_node_out, {symbolic::symbol("i_3")});
    }

    {
        auto& loop_i_4 = builder.add_for(
            root,
            symbolic::symbol("i_4"),
            symbolic::Lt(symbolic::symbol("i_4"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i_4"), symbolic::integer(1))
        );
        auto& body_i_4 = loop_i_4.root();
        auto& loop_j_4 = builder.add_for(
            body_i_4,
            symbolic::symbol("j_4"),
            symbolic::Lt(symbolic::symbol("j_4"), symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j_4"), symbolic::integer(1))
        );
        auto& body_j_4 = loop_j_4.root();

        auto& block = builder.add_block(body_j_4);

        auto& w_node_in = builder.add_access(block, "w");
        auto& w_node_out = builder.add_access(block, "w");
        auto& A_node = builder.add_access(block, "A");
        auto& x_node = builder.add_access(block, "x");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, w_node_in, tasklet, "_in3", {symbolic::symbol("i_4")});
        builder
            .add_computational_memlet(block, A_node, tasklet, "_in1", {symbolic::symbol("i_4"), symbolic::symbol("j_4")});
        builder.add_computational_memlet(block, x_node, tasklet, "_in2", {symbolic::symbol("j_4")});
        builder.add_computational_memlet(block, tasklet, "_out", w_node_out, {symbolic::symbol("i_4")});
    }

    sdfg::analysis::AnalysisManager manager(builder.subject());
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();
    auto outermost_loops = loop_analysis.outermost_loops();

    auto sdfg_copy = builder.subject().clone();

    sdfg::analysis::AnalysisManager manager_copy(*sdfg_copy);
    auto& loop_analysis_copy = manager_copy.get<analysis::LoopAnalysis>();
    auto outermost_loops_copy = loop_analysis_copy.outermost_loops();

    EXPECT_EQ(outermost_loops_copy.size(), 4);
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        EXPECT_EQ(
            sdfg::dyn_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops[i])->indvar()->get_name(),
            sdfg::dyn_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops_copy[i])->indvar()->get_name()
        );
    }

    auto sdfg_copy2 = builder.subject().clone();

    sdfg::analysis::AnalysisManager manager_copy2(*sdfg_copy2);
    auto& loop_analysis_copy2 = manager_copy2.get<analysis::LoopAnalysis>();
    auto outermost_loops_copy2 = loop_analysis_copy2.outermost_loops();

    EXPECT_EQ(outermost_loops_copy2.size(), 4);
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        EXPECT_EQ(
            sdfg::dyn_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops[i])->indvar()->get_name(),
            sdfg::dyn_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops_copy2[i])
                ->indvar()
                ->get_name()
        );
    }
}

// ---------------------------------------------------------------------------
// Incremental-update tests: removed_loop / moved_loop / copied_loop
// ---------------------------------------------------------------------------

namespace {

/**
 * Pointers to every loop of the reusable multi-nest fixture, in pre-order.
 * Pre-order over loops(): a0, a1, a2, b0, b1, b2, c0, c1, c2, c3 (indices 0..9).
 */
struct MultiNestLoops {
    // Nest A (idx 0..2): perfectly nested chain of two maps followed by a for-loop.
    structured_control_flow::Map* a0;
    structured_control_flow::Map* a1;
    structured_control_flow::For* a2;
    // Nest B (idx 3..5): a chain of 3 maps, b1 is non-perfectly nested due to an empty block in its body.
    structured_control_flow::Map* b0;
    structured_control_flow::Map* b1;
    structured_control_flow::Map* b2;
    // Nest C (idx 6..9): a map c0 with two child loops (c1, c2); one leaf map and one map -> for.
    structured_control_flow::Map* c0;
    structured_control_flow::Map* c1;
    structured_control_flow::Map* c2;
    structured_control_flow::For* c3;
};

/**
 * Builds three independent loop nests at the SDFG root, mixing maps and fors,
 * a non-perfectly-nested loop (empty block in a non-leaf body) and a loop with
 * multiple children. Reuse this across the incremental-update tests and then
 * issue a single removed_loop / moved_loop / copied_loop call.
 */
class MultiNestBuilder {
public:
    builder::StructuredSDFGBuilder& builder;
    MultiNestBuilder(builder::StructuredSDFGBuilder& builder) : builder(builder) {}

    ScheduleType sched_ = ScheduleType_Sequential::create();
    Sequence& root = builder.subject().root();

    Map& add_map(Sequence& parent, const std::string& iv) {
        builder.add_container(iv, types::Scalar(types::PrimitiveType::Int32));
        auto sym = symbolic::symbol(iv);
        return builder.add_map(
            parent,
            sym,
            symbolic::Lt(sym, symbolic::symbol("N")),
            symbolic::zero(),
            symbolic::add(sym, symbolic::one()),
            sched_
        );
    }
    For& add_for(Sequence& parent, const std::string& iv) {
        builder.add_container(iv, types::Scalar(types::PrimitiveType::Int32));
        auto sym = symbolic::symbol(iv);
        return builder.add_for(
            parent, sym, symbolic::Lt(sym, symbolic::symbol("N")), symbolic::zero(), symbolic::add(sym, symbolic::one())
        );
    }

    MultiNestLoops build_default_multi_nest() {
        MultiNestLoops loops;

        // Nest A: a0 -> a1 -> a2 (perfectly nested maps).
        auto& a0 = add_map(root, "i_a0");
        auto& a1 = add_map(a0.root(), "i_a1");
        auto& a2 = add_for(a1.root(), "i_a2");
        loops.a0 = &a0;
        loops.a1 = &a1;
        loops.a2 = &a2;

        // Nest B: for b0 with an empty block (=> non-perfectly nested) then b1 -> b2 (maps).
        auto& b0 = add_map(root, "i_b0");
        auto& b1 = add_map(b0.root(), "i_b1");
        builder.add_block(b1.root());
        auto& b2 = add_map(b1.root(), "i_b2");
        loops.b0 = &b0;
        loops.b1 = &b1;
        loops.b2 = &b2;

        // Nest C: c0 with two children: leaf map c1 and c2 -> for c3.
        auto& c0 = add_map(root, "i_c0");
        auto& c1 = add_map(c0.root(), "i_c1");
        auto& c2 = add_map(c0.root(), "i_c2");
        auto& c3 = add_for(c2.root(), "i_c3");
        loops.c0 = &c0;
        loops.c1 = &c1;
        loops.c2 = &c2;
        loops.c3 = &c3;

        return loops;
    }
};

MultiNestLoops build_multi_nest(builder::StructuredSDFGBuilder& builder) {
    MultiNestBuilder b(builder);
    return b.build_default_multi_nest();
}

/**
 * Verifies the core LoopAnalysis index invariant after an incremental update:
 *  - loop_id equals the loop's position in the pre-order loops() vector,
 *  - the subtree occupies a contiguous index range [loop_id, last_child_id],
 *  - last_child_id == loop_id + number-of-descendants.
 */
void verify_loop_index_consistency(analysis::LoopAnalysis& la) {
    const auto& order = la.loops_in_pre_order();
    for (uint32_t i = 0; i < order.size(); ++i) {
        auto* loop = order[i];
        const auto& info = la.loop_info_local(loop);
        EXPECT_EQ(info.loop_id, i) << "loop_id must match pre-order position";

        auto desc = la.descendants(loop);
        EXPECT_EQ(info.last_child_id, info.loop_id + desc.size()) << "last_child_id must cover exactly the subtree";
        for (uint32_t j = info.loop_id + 1; j <= info.last_child_id; ++j) {
            ASSERT_LT(j, order.size());
            EXPECT_EQ(desc.count(order[j]), 1u) << "index " << j << " in subtree range must be a descendant";
        }
    }
}

} // namespace

TEST(LoopAnalysisTest, RemovedLoop_Leaf) {
    builder::StructuredSDFGBuilder builder("sdfg_remove_leaf", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();
    ASSERT_EQ(la.loops_in_pre_order().size(), 10u);
    EXPECT_FALSE(la.loop_info(loops.a0).is_perfectly_parallel);

    dump_loop_info(la, "0.org");

    // Remove the innermost leaf of nest A.
    la.removed_loop(loops.a2);

    dump_loop_info(la, "1.remove");

    EXPECT_EQ(la.loops_in_pre_order().size(), 9u);
    for (auto* l : la.loops_in_pre_order()) {
        EXPECT_NE(l, loops.a2);
    }
    // a1 is now a leaf, a0 spans {a0, a1}.
    EXPECT_EQ(la.children(loops.a1).size(), 0u);
    EXPECT_EQ(la.loop_info_local(loops.a0).loop_id, 0u);
    EXPECT_EQ(la.loop_info_local(loops.a0).last_child_id, 1u);
    // Everything after the removal shifted down by one (c0 was 6, now 5).
    EXPECT_EQ(la.loop_info_local(loops.c0).loop_id, 5u);
    EXPECT_EQ(la.loop_info_local(loops.c0).last_child_id, 8u);
    EXPECT_TRUE(la.loop_info(loops.a0).is_perfectly_parallel);


    verify_loop_index_consistency(la);
    manager.invalidate_all();
}

TEST(LoopAnalysisTest, RemovedLoop_InternalSubtree) {
    builder::StructuredSDFGBuilder builder("sdfg_remove_subtree", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();
    ASSERT_EQ(la.loops_in_pre_order().size(), 10u);
    EXPECT_FALSE(la.loop_info(loops.c0).is_perfectly_parallel);
    EXPECT_FALSE(la.loop_info(loops.c0).is_perfectly_nested);

    dump_loop_info(la, "0.org");

    // Remove c2, which carries its child c3 along with it.
    la.removed_loop(loops.c2);

    dump_loop_info(la, "1.remove");

    EXPECT_EQ(la.loops_in_pre_order().size(), 8u);
    for (auto* l : la.loops_in_pre_order()) {
        EXPECT_NE(l, loops.c2);
        EXPECT_NE(l, loops.c3);
    }
    // c0 now has only c1 as a child.
    const auto& c0_children = la.children(loops.c0);
    EXPECT_EQ(c0_children.size(), 1u);
    EXPECT_EQ(c0_children.at(0), loops.c1);
    EXPECT_EQ(la.loop_info_local(loops.c0).last_child_id, la.loop_info_local(loops.c1).loop_id);
    EXPECT_TRUE(la.loop_info(loops.c0).is_perfectly_parallel);
    EXPECT_TRUE(la.loop_info(loops.c0).is_perfectly_nested);

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, RemovedLoop_WholeOutermostNest) {
    builder::StructuredSDFGBuilder builder("sdfg_remove_nest", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();
    ASSERT_EQ(la.outermost_loops().size(), 3u);

    // Remove the entire middle nest (for b0 plus b1 -> b2).
    la.removed_loop(loops.b0);

    EXPECT_EQ(la.loops_in_pre_order().size(), 7u);
    for (auto* l : la.loops_in_pre_order()) {
        EXPECT_NE(l, loops.b0);
        EXPECT_NE(l, loops.b1);
        EXPECT_NE(l, loops.b2);
    }
    EXPECT_EQ(la.outermost_loops().size(), 2u);
    // Nest C shifted to directly follow nest A (c0 was 6, now 3).
    EXPECT_EQ(la.loop_info_local(loops.c0).loop_id, 3u);
    EXPECT_EQ(la.loop_info_local(loops.c0).last_child_id, 6u);

    verify_loop_index_consistency(la);
}

// The moved_loop / copied_loop APIs are still under construction. These document
// the intended scenarios and post-conditions; enable once the methods are implemented.

TEST(LoopAnalysisTest, MovedLoop_Reparent) {
    builder::StructuredSDFGBuilder builder("sdfg_move", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    EXPECT_FALSE(la.loop_info(loops.c0).is_perfectly_parallel);
    EXPECT_FALSE(la.loop_info(loops.c0).is_perfectly_nested);
    EXPECT_TRUE(la.loop_info(loops.a2).is_perfectly_nested);

    // Move c2 (with its child c3) from under c0 to become a child of a2.
    la.moved_loop(loops.c2, loops.a2);

    EXPECT_EQ(la.parent_loop(loops.c2), loops.a2);
    const auto& a2_children = la.children(loops.a2);
    EXPECT_NE(std::find(a2_children.begin(), a2_children.end(), loops.c2), a2_children.end());
    const auto& c0_children = la.children(loops.c0);
    EXPECT_EQ(std::find(c0_children.begin(), c0_children.end(), loops.c2), c0_children.end());

    EXPECT_TRUE(la.loop_info(loops.c0).is_perfectly_parallel);
    EXPECT_TRUE(la.loop_info(loops.c0).is_perfectly_nested);
    EXPECT_TRUE(la.loop_info(loops.a2).is_perfectly_nested);

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, MovedLoop_ToOwnParent) {
    builder::StructuredSDFGBuilder builder("sdfg_move", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    // move to a parent
    la.moved_loop(loops.a2, loops.a0);

    EXPECT_EQ(la.parent_loop(loops.a2), loops.a0);
    const auto& a0_children = la.children(loops.a0);
    EXPECT_NE(std::find(a0_children.begin(), a0_children.end(), loops.a2), a0_children.end());
    const auto& a1_children = la.children(loops.a1);
    EXPECT_EQ(std::find(a1_children.begin(), a1_children.end(), loops.a2), a1_children.end());

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, Added_LocalContents) {
    builder::StructuredSDFGBuilder builder("sdfg_move", FunctionType_CPU);
    MultiNestBuilder b(builder);

    auto a0 = &b.add_map(b.root, "i_a0");
    auto a1 = &b.add_map(a0->root(), "i_a1");
    auto a2 = &b.add_map(a1->root(), "i_a2");

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    dump_loop_info(la, "0.org");

    EXPECT_TRUE(la.loop_info(a0).is_perfectly_nested);
    EXPECT_TRUE(la.loop_info(a0).is_perfectly_parallel);
    EXPECT_EQ(la.loop_info(a0).map_stack_depth, 3u);
    EXPECT_EQ(la.loop_info(a1).map_stack_depth, 2u);
    EXPECT_EQ(la.loop_info(a2).map_stack_depth, 1u);
    EXPECT_FALSE(la.loop_info(a0).has_side_effects);

    // move to a parent
    la.added_local_contents(a1, true, true);

    dump_loop_info(la, "0.added");

    EXPECT_EQ(la.parent_loop(a2), a1);
    EXPECT_EQ(la.parent_loop(a1), a0);

    EXPECT_TRUE(la.loop_info(a2).is_perfectly_nested);
    EXPECT_FALSE(la.loop_info(a1).is_perfectly_nested);
    EXPECT_FALSE(la.loop_info(a0).is_perfectly_nested);

    EXPECT_TRUE(la.loop_info(a2).is_perfectly_parallel);
    EXPECT_TRUE(la.loop_info(a1).is_perfectly_parallel);
    EXPECT_TRUE(la.loop_info(a0).is_perfectly_parallel);

    EXPECT_FALSE(la.loop_info(a2).has_side_effects);
    EXPECT_TRUE(la.loop_info(a1).has_side_effects);
    EXPECT_TRUE(la.loop_info(a0).has_side_effects);


    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, CopiedLoop_Append) {
    builder::StructuredSDFGBuilder builder("sdfg_copy", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    EXPECT_FALSE(la.loop_info(loops.b0).is_perfectly_nested);
    EXPECT_TRUE(la.loop_info(loops.b0).is_perfectly_parallel);

    // Materialize a copy of a2 as a new child of b1, then register it.
    auto sym = symbolic::symbol("i_copy");
    builder.add_container("i_copy", types::Scalar(types::PrimitiveType::Int32));
    auto& new_loop = builder.add_for(
        loops.b1->root(),
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(sym, symbolic::one())
    );

    la.copied_loop(loops.a2, loops.b1, &new_loop, true);

    // old one still there
    EXPECT_EQ(la.parent_loop(loops.a2), loops.a1);
    const auto& a1_children = la.children(loops.a1);
    EXPECT_NE(std::find(a1_children.begin(), a1_children.end(), loops.a2), a1_children.end());

    EXPECT_EQ(la.parent_loop(&new_loop), loops.b1);
    const auto& b1_children = la.children(loops.b1);
    EXPECT_NE(std::find(b1_children.begin(), b1_children.end(), &new_loop), b1_children.end());

    EXPECT_FALSE(la.loop_info(loops.b0).is_perfectly_nested);
    EXPECT_FALSE(la.loop_info(loops.b0).is_perfectly_parallel);

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, CopiedLoop_AppendBack) {
    builder::StructuredSDFGBuilder builder("sdfg_copy_back", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    ASSERT_EQ(la.loops_in_pre_order().size(), 10u);

    // Append a fresh map as the last child of c0 (which already has c1, c2).
    auto sym = symbolic::symbol("i_copy_back");
    builder.add_container("i_copy_back", types::Scalar(types::PrimitiveType::Int32));
    auto& new_loop = builder.add_map(
        loops.c0->root(),
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(sym, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    la.copied_loop(loops.c1, loops.c0, &new_loop); // default: append at the back

    EXPECT_EQ(la.loops_in_pre_order().size(), 11u);
    EXPECT_EQ(la.parent_loop(&new_loop), loops.c0);

    const auto& c0_children = la.children(loops.c0);
    ASSERT_EQ(c0_children.size(), 3u);
    EXPECT_EQ(c0_children.back(), &new_loop); // appended after the existing children

    // The new loop is the last entry in pre-order; c0's subtree grew to include it.
    EXPECT_EQ(la.loop_info_local(&new_loop).loop_id, 10u);
    EXPECT_EQ(la.loop_info_local(loops.c0).last_child_id, 10u);

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, CopiedLoop_Subtree) {
    builder::StructuredSDFGBuilder builder("sdfg_copy_subtree", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    // Materialize a small nested subtree (map -> for) as a new child of the leaf a2.
    auto outer = symbolic::symbol("i_cp_outer");
    auto inner = symbolic::symbol("i_cp_inner");
    builder.add_container("i_cp_outer", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i_cp_inner", types::Scalar(types::PrimitiveType::Int32));
    auto& new_outer = builder.add_map(
        loops.a2->root(),
        outer,
        symbolic::Lt(outer, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(outer, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& new_inner = builder.add_for(
        new_outer.root(),
        inner,
        symbolic::Lt(inner, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(inner, symbolic::one())
    );

    la.copied_loop(loops.c2, loops.a2, &new_outer); // a2 had no children -> back == front

    EXPECT_EQ(la.loops_in_pre_order().size(), 12u);
    // The whole copied subtree is registered with correct parentage.
    EXPECT_EQ(la.parent_loop(&new_outer), loops.a2);
    EXPECT_EQ(la.parent_loop(&new_inner), &new_outer);

    // Pre-order: a2 (idx 2) is now immediately followed by the new subtree.
    EXPECT_EQ(la.loop_info_local(&new_outer).loop_id, 3u);
    EXPECT_EQ(la.loop_info_local(&new_inner).loop_id, 4u);
    EXPECT_EQ(la.loop_info_local(&new_outer).last_child_id, 4u);

    // Nest aggregates of the freshly added subtree.
    EXPECT_EQ(la.loop_info(&new_outer).num_loops, 2u);
    EXPECT_EQ(la.loop_info(&new_outer).num_maps, 1u);
    EXPECT_EQ(la.loop_info(&new_outer).num_fors, 1u);
    EXPECT_EQ(la.loop_info(&new_outer).max_depth, 2u);
    EXPECT_EQ(la.loop_info(&new_outer).loop_level, 3u); // a2 sits at level 2

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, CopiedLoop_AsOutermost) {
    builder::StructuredSDFGBuilder builder("sdfg_copy_outer", FunctionType_CPU);
    auto loops = build_multi_nest(builder);

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    ASSERT_EQ(la.outermost_loops().size(), 3u);

    // Append a fresh outermost map directly at the SDFG root (no parent loop).
    auto sym = symbolic::symbol("i_cp_root");
    builder.add_container("i_cp_root", types::Scalar(types::PrimitiveType::Int32));
    auto& new_loop = builder.add_map(
        builder.subject().root(),
        sym,
        symbolic::Lt(sym, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(sym, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    la.copied_loop(loops.a0, nullptr, &new_loop); // new outermost nest, appended at the back

    EXPECT_EQ(la.loops_in_pre_order().size(), 11u);
    EXPECT_TRUE(la.is_outermost_loop(&new_loop));
    EXPECT_EQ(la.parent_loop(&new_loop), nullptr);

    const auto& outer = la.outermost_loops();
    ASSERT_EQ(outer.size(), 4u);
    EXPECT_EQ(outer.back(), &new_loop);
    EXPECT_EQ(la.loop_info_local(&new_loop).loop_id, 10u);
    EXPECT_EQ(la.loop_info(&new_loop).loopnest_index, 3); // 4th outermost nest

    verify_loop_index_consistency(la);
}

TEST(LoopAnalysisTest, MapStackDepth_3outers) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    MultiNestBuilder b(builder);

    auto& a0 = b.add_map(b.root, "i0");
    auto& a1 = b.add_map(a0.root(), "i1");
    auto& a2 = b.add_map(a1.root(), "i2");
    auto& a3 = b.add_for(a2.root(), "i3");
    auto& a4 = b.add_for(a3.root(), "i4");
    auto& a5 = b.add_for(a4.root(), "i5");

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(la.loop_info(&a5).map_stack_depth, 0);
    EXPECT_EQ(la.loop_info(&a4).map_stack_depth, 0);
    EXPECT_EQ(la.loop_info(&a3).map_stack_depth, 0);
    EXPECT_EQ(la.loop_info(&a2).map_stack_depth, 1);
    EXPECT_EQ(la.loop_info(&a1).map_stack_depth, 2);
    EXPECT_EQ(la.loop_info(&a0).map_stack_depth, 3);
}

TEST(LoopAnalysisTest, MapStackDepth_just2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    MultiNestBuilder b(builder);

    auto& a0 = b.add_map(b.root, "i0");
    auto& a1 = b.add_map(a0.root(), "i1");

    analysis::AnalysisManager manager(builder.subject());
    auto& la = manager.get<analysis::LoopAnalysis>();

    EXPECT_EQ(la.loop_info(&a1).map_stack_depth, 1);
    EXPECT_EQ(la.loop_info(&a0).map_stack_depth, 2);
}
