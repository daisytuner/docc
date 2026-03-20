#include "sdfg/transformations/out_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(OutLocalStorage, Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(4);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, "C");
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check
    EXPECT_EQ(new_root.size(), 3);
    auto init_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(0).first);
    EXPECT_NE(init_block, nullptr);
    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    bool c_access = false;
    bool a_access = false;
    for (auto& node : init_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : init_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        }
    }

    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    EXPECT_EQ(loop_block->dataflow().nodes().size(), 4);
    EXPECT_EQ(loop_block->dataflow().edges().size(), 3);
    int accesses = 0;
    a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A") {
            a_access = true;
        } else if (access_node->data() == "__daisy_out_local_storage_C") {
            accesses++;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    auto deinit_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(2).first);
    EXPECT_NE(deinit_block, nullptr);

    EXPECT_EQ(deinit_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(deinit_block->dataflow().edges().size(), 2);
    c_access = false;
    a_access = false;
    for (auto& node : deinit_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : deinit_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
        }
    }
}

TEST(OutLocalStorage, Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc(base_desc, {symbolic::symbol("N")});
    builder.add_container("A", desc, true);
    builder.add_container("C", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(100);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, "C");
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check
    {
        EXPECT_EQ(new_root.size(), 3);
        auto init_for = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
        EXPECT_NE(init_for, nullptr);

        EXPECT_TRUE(symbolic::
                        eq(symbolic::subs(init_for->condition(), init_for->indvar(), loop.indvar()), loop.condition()));
        EXPECT_TRUE(symbolic::eq(symbolic::subs(init_for->update(), init_for->indvar(), loop.indvar()), loop.update()));
        EXPECT_TRUE(symbolic::eq(init_for->init(), loop.init()));

        auto& init_body = init_for->root();
        EXPECT_EQ(init_body.size(), 1);
        auto init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
        EXPECT_NE(init_block, nullptr);
        EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(init_block->dataflow().edges().size(), 2);
        bool c_access = false;
        bool a_access = false;
        for (auto& node : init_block->dataflow().nodes()) {
            if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
                if (access->data() == "C") {
                    c_access = true;
                } else if (access->data() == "__daisy_out_local_storage_C") {
                    a_access = true;
                }
            } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
                EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
            }
        }
        EXPECT_TRUE(a_access);
        EXPECT_TRUE(c_access);

        for (auto& memlet : init_block->dataflow().edges()) {
            EXPECT_EQ(memlet.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), init_for->indvar()));
            if (memlet.dst_conn() == "void") {
                auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
                EXPECT_NE(access, nullptr);
                EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
            } else if (memlet.src_conn() == "void") {
                auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
                EXPECT_NE(access, nullptr);
                EXPECT_EQ(access->data(), "C");
            }
        }
    }

    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    EXPECT_EQ(loop_block->dataflow().nodes().size(), 4);
    EXPECT_EQ(loop_block->dataflow().edges().size(), 3);
    int accesses = 0;
    bool a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A") {
            a_access = true;
        } else if (access_node->data() == "__daisy_out_local_storage_C") {
            accesses++;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    {
        auto init_for = dynamic_cast<structured_control_flow::For*>(&new_root.at(2).first);
        EXPECT_NE(init_for, nullptr);

        EXPECT_TRUE(symbolic::
                        eq(symbolic::subs(init_for->condition(), init_for->indvar(), loop.indvar()), loop.condition()));
        EXPECT_TRUE(symbolic::eq(symbolic::subs(init_for->update(), init_for->indvar(), loop.indvar()), loop.update()));
        EXPECT_TRUE(symbolic::eq(init_for->init(), loop.init()));

        auto& init_body = init_for->root();
        EXPECT_EQ(init_body.size(), 1);
        auto init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
        EXPECT_NE(init_block, nullptr);
        EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(init_block->dataflow().edges().size(), 2);
        bool c_access = false;
        bool a_access = false;
        for (auto& node : init_block->dataflow().nodes()) {
            if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
                if (access->data() == "C") {
                    c_access = true;
                } else if (access->data() == "__daisy_out_local_storage_C") {
                    a_access = true;
                }
            } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
                EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
            }
        }
        EXPECT_TRUE(a_access);
        EXPECT_TRUE(c_access);

        for (auto& memlet : init_block->dataflow().edges()) {
            EXPECT_EQ(memlet.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), init_for->indvar()));
            if (memlet.dst_conn() == "void") {
                auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
                EXPECT_NE(access, nullptr);
                EXPECT_EQ(access->data(), "C");
            } else if (memlet.src_conn() == "void") {
                auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
                EXPECT_NE(access, nullptr);
                EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
            }
        }
    }
}

TEST(OutLocalStorage, Fail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "i");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::integer(0)}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply
    transformations::OutLocalStorage transformation(loop, "C");
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}
