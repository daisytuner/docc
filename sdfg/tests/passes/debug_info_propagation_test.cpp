#include "sdfg/passes/debug_info_propagation.h"

#include <gtest/gtest.h>

#include <ostream>

using namespace sdfg;

TEST(DebugInfoPropagationTest, BlockPropagation_Node) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 =
        builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"}, DebugInfo{"main.c", 1, 1, 1, 1});
    builder.add_computational_memlet(block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)});

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DebugInfoPropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(builder, analysis_manager);

    EXPECT_TRUE(builder.subject().root().debug_info().has());
    EXPECT_EQ(builder.subject().root().debug_info().filename(), "main.c");
    EXPECT_EQ(builder.subject().root().debug_info().start_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().start_column(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_column(), 1);
}

TEST(DebugInfoPropagationTest, BlockPropagation_Edge) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block1, node1_1, tasklet_1, "_in", {symbolic::integer(0)}, DebugInfo{"main.c", 1, 1, 1, 1}
    );
    builder.add_computational_memlet(
        block1, tasklet_1, "_out", node2_1, {symbolic::integer(0)}, DebugInfo{"main.c", 2, 2, 2, 2}
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DebugInfoPropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(builder, analysis_manager);

    EXPECT_TRUE(builder.subject().root().debug_info().has());
    EXPECT_EQ(builder.subject().root().debug_info().filename(), "main.c");
    EXPECT_EQ(builder.subject().root().debug_info().start_line(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().start_column(), 1);
    EXPECT_EQ(builder.subject().root().debug_info().end_line(), 2);
    EXPECT_EQ(builder.subject().root().debug_info().end_column(), 2);
}
