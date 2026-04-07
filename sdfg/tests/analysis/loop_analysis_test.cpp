#include "sdfg/analysis/loop_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
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
            dynamic_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops[i])->indvar()->get_name(),
            dynamic_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops_copy[i])->indvar()->get_name()
        );
    }

    auto sdfg_copy2 = builder.subject().clone();

    sdfg::analysis::AnalysisManager manager_copy2(*sdfg_copy2);
    auto& loop_analysis_copy2 = manager_copy2.get<analysis::LoopAnalysis>();
    auto outermost_loops_copy2 = loop_analysis_copy2.outermost_loops();

    EXPECT_EQ(outermost_loops_copy2.size(), 4);
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        EXPECT_EQ(
            dynamic_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops[i])->indvar()->get_name(),
            dynamic_cast<sdfg::structured_control_flow::StructuredLoop*>(outermost_loops_copy2[i])->indvar()->get_name()
        );
    }
}
