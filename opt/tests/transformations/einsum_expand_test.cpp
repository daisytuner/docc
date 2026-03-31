#include "sdfg/transformations/einsum_expand.h"

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
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
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

static bool subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2) {
    if (subset1.size() != subset2.size()) {
        return false;
    }
    for (size_t i = 0; i < subset1.size(); ++i) {
        if (!symbolic::eq(subset1[i], subset2[i])) {
            return false;
        }
    }
    return true;
}

TEST(EinsumExpandTest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block);
    auto& dfg = new_block->dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 3);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_einsum_node = dynamic_cast<einsum::EinsumNode*>(new_libnode);
    ASSERT_TRUE(new_einsum_node);
    EXPECT_EQ(new_einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(new_einsum_node->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    EXPECT_EQ(new_einsum_node->dims().size(), 1);
    ASSERT_GE(new_einsum_node->dims().size(), 1);
    EXPECT_TRUE(symbolic::eq(new_einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->init(0), init));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->bound(0), bound));

    EXPECT_EQ(new_einsum_node->in_indices().size(), 2);
    ASSERT_GE(new_einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(new_einsum_node->in_indices(0), {i}));
    EXPECT_TRUE(subsets_eq(new_einsum_node->in_indices(1), {i}));
    EXPECT_TRUE(subsets_eq(new_einsum_node->out_indices(), {i}));

    auto conn2cont = get_conn2cont(dfg, *new_einsum_node);
    EXPECT_TRUE(conn2cont.contains("_in"));
    EXPECT_EQ(conn2cont.at("_in"), "b");
    EXPECT_TRUE(conn2cont.contains("__einsum_out"));
    EXPECT_EQ(conn2cont.at("__einsum_out"), "a");
}

TEST(EinsumExpandTest, Multiple) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc(base_desc, symbolic::integer(20));
    types::Pointer desc2(desc);
    builder.add_container("a", desc2);
    builder.add_container("b", desc2);

    auto i = symbolic::symbol("i");
    auto init_i = symbolic::integer(3);
    auto bound_i = symbolic::integer(10);
    auto& for_node_i = builder.add_for(root, i, symbolic::Lt(i, bound_i), init_i, symbolic::add(i, symbolic::one()));

    auto j = symbolic::symbol("j");
    auto init_j = symbolic::integer(8);
    auto bound_j = symbolic::integer(20);
    auto& for_node_j =
        builder.add_for(for_node_i.root(), j, symbolic::Lt(j, bound_j), init_j, symbolic::add(j, symbolic::one()));

    auto& block = builder.add_block(for_node_j.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i, j}, {{i, j}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation1(einsum_node);
    ASSERT_TRUE(transformation1.can_be_applied(builder, analysis_manager));
    transformation1.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(for_node_i.root().size(), 1);
    ASSERT_GE(for_node_i.root().size(), 1);

    auto* intermediate_block = dynamic_cast<structured_control_flow::Block*>(&for_node_i.root().at(0).first);
    ASSERT_TRUE(intermediate_block);
    auto& intermediate_dfg = intermediate_block->dataflow();
    EXPECT_EQ(intermediate_dfg.data_nodes().size(), 3);
    EXPECT_EQ(intermediate_dfg.tasklets().size(), 0);
    EXPECT_EQ(intermediate_dfg.library_nodes().size(), 1);
    ASSERT_GE(intermediate_dfg.library_nodes().size(), 1);

    auto* intermediate_libnode = *intermediate_dfg.library_nodes().begin();
    auto* intermediate_einsum_node = dynamic_cast<einsum::EinsumNode*>(intermediate_libnode);
    ASSERT_TRUE(intermediate_einsum_node);

    transformations::EinsumExpand transformation2(*intermediate_einsum_node);
    ASSERT_TRUE(transformation2.can_be_applied(builder, analysis_manager));
    transformation2.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block);
    auto& dfg = new_block->dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 3);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_einsum_node = dynamic_cast<einsum::EinsumNode*>(new_libnode);
    ASSERT_TRUE(new_einsum_node);
    EXPECT_EQ(new_einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(new_einsum_node->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    EXPECT_EQ(new_einsum_node->dims().size(), 2);
    ASSERT_GE(new_einsum_node->dims().size(), 2);
    EXPECT_TRUE(symbolic::eq(new_einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->init(0), init_i));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->bound(0), bound_i));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->indvar(1), j));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->init(1), init_j));
    EXPECT_TRUE(symbolic::eq(new_einsum_node->bound(1), bound_j));

    EXPECT_EQ(new_einsum_node->in_indices().size(), 2);
    ASSERT_GE(new_einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(new_einsum_node->in_indices(0), {i, j}));
    EXPECT_TRUE(subsets_eq(new_einsum_node->in_indices(1), {i, j}));
    EXPECT_TRUE(subsets_eq(new_einsum_node->out_indices(), {i, j}));

    auto conn2cont = get_conn2cont(dfg, *new_einsum_node);
    EXPECT_TRUE(conn2cont.contains("_in"));
    EXPECT_EQ(conn2cont.at("_in"), "b");
    EXPECT_TRUE(conn2cont.contains("__einsum_out"));
    EXPECT_EQ(conn2cont.at("__einsum_out"), "a");
}

TEST(EinsumExpandTest, DataFlowBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c_in = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in", {i});
    builder.add_computational_memlet(block, tasklet, "_out", b, {i});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, DataFlowAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_out = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a, {}, desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {i});
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i});

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, ControlFlowBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block1 = builder.add_block(for_node.root());
    auto& c_in = builder.add_access(block1, "c");
    auto& b_out = builder.add_access(block1, "b");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, c_in, tasklet, "_in", {i});
    builder.add_computational_memlet(block1, tasklet, "_out", b_out, {i});

    auto& block2 = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block2, "a");
    auto& a_out = builder.add_access(block2, "a");
    auto& b_in = builder.add_access(block2, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block2, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block2, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block2, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block2, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, ControlFlowAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block1 = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block1, "a");
    auto& a_out = builder.add_access(block1, "a");
    auto& b_in = builder.add_access(block1, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block1, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block1, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block1, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block1, libnode, "__einsum_out", a_out, {}, desc);

    auto& block2 = builder.add_block(for_node.root());
    auto& a_in2 = builder.add_access(block2, "a");
    auto& c_out = builder.add_access(block2, "b");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, a_in2, tasklet, "_in", {i});
    builder.add_computational_memlet(block2, tasklet, "_out", c_out, {i});

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, InsufficientLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::integer(2)));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, LoopCarriedDependency) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{symbolic::symbol("j")}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, a_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, LocalWriteBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", base_desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c_in = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in", {i});
    builder.add_computational_memlet(block, tasklet, "_out", b, {});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {i}, {{}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, LocalReadAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", base_desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto& block = builder.add_block(for_node.root());
    auto& a_in = builder.add_access(block, "a");
    auto& a = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_out = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {}, {}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a, {}, desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i});

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumExpandTest, LocalSymbols) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("n", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto i = symbolic::symbol("i");
    auto init = symbolic::integer(3);
    auto bound = symbolic::integer(10);
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::one()));

    auto j = symbolic::symbol("j");
    auto zero = symbolic::zero();
    auto n = symbolic::symbol("n");

    auto& block = builder.add_block(for_node.root());
    auto& i_in = builder.add_access(block, "i");
    auto& constant_five = builder.add_constant(block, "5", sym_desc);
    auto& n_out = builder.add_access(block, "n");
    auto& a_in = builder.add_access(block, "a");
    auto& a_out = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, i_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, constant_five, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", n_out, {});
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {{j, zero, n}}, {i}, {{i}});
    builder.add_computational_memlet(block, a_in, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block, b_in, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", a_out, {}, desc);

    auto& einsum_node = static_cast<einsum::EinsumNode&>(libnode);
    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumExpand transformation(einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}
