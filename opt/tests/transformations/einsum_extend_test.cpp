#include "sdfg/transformations/einsum_extend.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

static std::unordered_map<std::string, std::string>
get_conn2cont(data_flow::DataFlowGraph& dfg, data_flow::DataFlowNode& node) {
    std::unordered_map<std::string, std::string> result;
    for (auto& iedge : dfg.in_edges(node)) {
        if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src())) {
            result.insert({iedge.dst_conn(), access_node->data()});
        }
    }
    return result;
}

TEST(EinsumExtendTest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("tmp", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_in = builder.add_access(block, "c");
    auto& tmp = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, c_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", tmp, {});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {}, {{}});
    builder.add_computational_memlet(block, tmp, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExtend transformation(einsum_node);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 4);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_einsum_node = dynamic_cast<einsum::EinsumNode*>(new_libnode);
    ASSERT_TRUE(new_einsum_node);
    EXPECT_EQ(new_einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(new_einsum_node->inputs(), std::vector<std::string>({"_in_in1", "_in_in2", "__einsum_out"}));

    auto conn2cont = get_conn2cont(dfg, *new_libnode);
    EXPECT_TRUE(conn2cont.contains("_in_in1"));
    EXPECT_EQ(conn2cont.at("_in_in1"), "b");
    EXPECT_TRUE(conn2cont.contains("_in_in2"));
    EXPECT_EQ(conn2cont.at("_in_in2"), "c");
    EXPECT_TRUE(conn2cont.contains("__einsum_out"));
    EXPECT_EQ(conn2cont.at("__einsum_out"), "a");
}

TEST(EinsumExtendTest, Multiple) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("e", desc);
    builder.add_container("f", desc);
    builder.add_container("tmp1", desc);
    builder.add_container("tmp2", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_in = builder.add_access(block, "c");
    auto& d_in = builder.add_access(block, "d");
    auto& e_in = builder.add_access(block, "e");
    auto& f_in = builder.add_access(block, "f");
    auto& tmp1 = builder.add_access(block, "tmp1");
    auto& tmp2 = builder.add_access(block, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, d_in, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", tmp1, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, e_in, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, f_in, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", tmp2, {});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in1", "_in2", "_in3"}, {}, {}, {{}, {}, {}});
    builder.add_computational_memlet(block, tmp1, libnode, "_in1", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in2", {}, desc);
    builder.add_computational_memlet(block, tmp2, libnode, "_in3", {}, desc);
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExtend transformation(einsum_node);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 7);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_einsum_node = dynamic_cast<einsum::EinsumNode*>(new_libnode);
    ASSERT_TRUE(new_einsum_node);
    EXPECT_EQ(new_einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(
        new_einsum_node->inputs(),
        std::vector<std::string>({"_in1_in1", "_in1_in2", "_in2", "_in3_in1", "_in3_in2", "__einsum_out"})
    );

    auto conn2cont = get_conn2cont(dfg, *new_libnode);
    EXPECT_TRUE(conn2cont.contains("_in1_in1"));
    EXPECT_EQ(conn2cont.at("_in1_in1"), "c");
    EXPECT_TRUE(conn2cont.contains("_in1_in2"));
    EXPECT_EQ(conn2cont.at("_in1_in2"), "d");
    EXPECT_TRUE(conn2cont.contains("_in2"));
    EXPECT_EQ(conn2cont.at("_in2"), "b");
    EXPECT_TRUE(conn2cont.contains("_in3_in1"));
    EXPECT_EQ(conn2cont.at("_in3_in1"), "e");
    EXPECT_TRUE(conn2cont.contains("_in3_in2"));
    EXPECT_EQ(conn2cont.at("_in3_in2"), "f");
    EXPECT_TRUE(conn2cont.contains("__einsum_out"));
    EXPECT_EQ(conn2cont.at("__einsum_out"), "a");
}

TEST(EinsumExtendTest, MultipleAfterEachOther) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("tmp1", desc);
    builder.add_container("tmp2", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_in = builder.add_access(block, "c");
    auto& d_in = builder.add_access(block, "d");
    auto& tmp1 = builder.add_access(block, "tmp1");
    auto& tmp2 = builder.add_access(block, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b_in, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, c_in, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", tmp1, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, tmp1, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, d_in, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", tmp2, {});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {}, {{}});
    builder.add_computational_memlet(block, tmp2, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExtend transformation1(einsum_node);
    ASSERT_TRUE(transformation1.can_be_applied(builder, analysis_manager));
    transformation1.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 6);
    EXPECT_EQ(dfg.tasklets().size(), 1);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* intermediate_libnode = *dfg.library_nodes().begin();
    auto* intermeidate_einsum_node = dynamic_cast<einsum::EinsumNode*>(intermediate_libnode);
    ASSERT_TRUE(intermeidate_einsum_node);

    transformations::EinsumExtend transformation2(*intermeidate_einsum_node);
    ASSERT_TRUE(transformation2.can_be_applied(builder, analysis_manager));
    transformation2.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(dfg.data_nodes().size(), 5);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_einsum_node = dynamic_cast<einsum::EinsumNode*>(new_libnode);
    ASSERT_TRUE(new_einsum_node);
    EXPECT_EQ(new_einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(
        new_einsum_node->inputs(), std::vector<std::string>({"_in_in1_in1", "_in_in1_in2", "_in_in2", "__einsum_out"})
    );

    auto conn2cont = get_conn2cont(dfg, *new_libnode);
    EXPECT_TRUE(conn2cont.contains("_in_in1_in1"));
    EXPECT_EQ(conn2cont.at("_in_in1_in1"), "b");
    EXPECT_TRUE(conn2cont.contains("_in_in1_in2"));
    EXPECT_EQ(conn2cont.at("_in_in1_in2"), "c");
    EXPECT_TRUE(conn2cont.contains("_in_in2"));
    EXPECT_EQ(conn2cont.at("_in_in2"), "d");
    EXPECT_TRUE(conn2cont.contains("__einsum_out"));
    EXPECT_EQ(conn2cont.at("__einsum_out"), "a");
}
