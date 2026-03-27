#include "sdfg/einsum/einsum.h"

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(EinsumNodeTest, SimpleGEMM) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add block with EinsumNode for GEMM
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block, DebugInfo(), {"_in1", "_in2"}, {{i, zero, l}, {j, zero, m}, {k, zero, n}}, {i, j}, {{i, k}, {k, j}}
    );
    builder.add_computational_memlet(block, A, libnode, "_in1", {}, desc_n);
    builder.add_computational_memlet(block, B, libnode, "_in2", {}, desc_m);
    builder.add_computational_memlet(block, C1, libnode, "__einsum_out", {}, desc_m);
    builder.add_computational_memlet(block, libnode, "__einsum_out", C2, {}, desc_m);

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    EXPECT_EQ(einsum_node->dims().size(), 3);
    ASSERT_GE(einsum_node->dims().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(0), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(0), l));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(1), j));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(1), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(1), m));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(2), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(2), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(2), n));

    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->out_indices().size(), 2);
    ASSERT_GE(einsum_node->out_indices().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(1), j));

    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 3);
    ASSERT_GE(einsum_node->in_indices().size(), 3);
    EXPECT_EQ(einsum_node->in_indices(0).size(), 2);
    ASSERT_GE(einsum_node->in_indices(0).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 1), k));
    EXPECT_EQ(einsum_node->in_indices(1).size(), 2);
    ASSERT_GE(einsum_node->in_indices(1).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 0), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 1), j));
    EXPECT_EQ(einsum_node->in_indices(2).size(), 2);
    ASSERT_GE(einsum_node->in_indices(2).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 1), j));

    auto symbols = einsum_node->symbols();
    EXPECT_EQ(symbols.size(), 6);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(k));
    EXPECT_TRUE(symbols.contains(l));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(n));

    EXPECT_EQ(
        einsum_node->toStr(),
        "__einsum_out[i][j] = _in1[i][k] * _in2[k][j] + __einsum_out[i][j] for i = 0 : l for j = 0 : m for k = 0 : n"
    );

    EXPECT_TRUE(symbolic::eq(einsum_node->flop(), SymEngine::mul({l, m, n, symbolic::integer(2)})));

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(EinsumNodeTest, ExpandGEMM) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add block with EinsumNode for GEMM
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block, DebugInfo(), {"_in1", "_in2"}, {{i, zero, l}, {j, zero, m}, {k, zero, n}}, {i, j}, {{i, k}, {k, j}}
    );
    builder.add_computational_memlet(block, A, libnode, "_in1", {}, desc_n);
    builder.add_computational_memlet(block, B, libnode, "_in2", {}, desc_m);
    builder.add_computational_memlet(block, C1, libnode, "__einsum_out", {}, desc_m);
    builder.add_computational_memlet(block, libnode, "__einsum_out", C2, {}, desc_m);

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(einsum_node->expand(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* loop1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(loop1);
    // Indvars are renamed during expansion for uniqueness, so get them from the loops
    auto i_renamed = loop1->indvar();
    EXPECT_TRUE(symbolic::eq(loop1->init(), zero));
    EXPECT_TRUE(symbolic::eq(loop1->condition(), symbolic::Lt(i_renamed, l)));
    EXPECT_TRUE(symbolic::eq(loop1->update(), symbolic::add(i_renamed, symbolic::one())));
    EXPECT_EQ(loop1->root().size(), 1);
    ASSERT_GE(loop1->root().size(), 1);

    auto* loop2 = dynamic_cast<structured_control_flow::Map*>(&loop1->root().at(0).first);
    ASSERT_TRUE(loop2);
    auto j_renamed = loop2->indvar();
    EXPECT_TRUE(symbolic::eq(loop2->init(), zero));
    EXPECT_TRUE(symbolic::eq(loop2->condition(), symbolic::Lt(j_renamed, m)));
    EXPECT_TRUE(symbolic::eq(loop2->update(), symbolic::add(j_renamed, symbolic::one())));
    EXPECT_EQ(loop2->root().size(), 1);
    ASSERT_GE(loop2->root().size(), 1);

    auto* loop3 = dynamic_cast<structured_control_flow::For*>(&loop2->root().at(0).first);
    ASSERT_TRUE(loop3);
    auto k_renamed = loop3->indvar();
    EXPECT_TRUE(symbolic::eq(loop3->init(), zero));
    EXPECT_TRUE(symbolic::eq(loop3->condition(), symbolic::Lt(k_renamed, n)));
    EXPECT_TRUE(symbolic::eq(loop3->update(), symbolic::add(k_renamed, symbolic::one())));
    EXPECT_EQ(loop3->root().size(), 1);
    ASSERT_GE(loop3->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&loop3->root().at(0).first);
    ASSERT_TRUE(new_block);
    auto& dfg = new_block->dataflow();
    EXPECT_EQ(dfg.tasklets().size(), 1);
    auto* tasklet = *dfg.tasklets().begin();
    ASSERT_TRUE(tasklet);
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_fma);
    EXPECT_EQ(tasklet->output(), "_out");
    for (auto& oedge : dfg.out_edges(*tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 2);
        ASSERT_GE(oedge.subset().size(), 2);
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), i_renamed));
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(1), j_renamed));
        EXPECT_EQ(access_node->data(), "C");
    }
    EXPECT_EQ(tasklet->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
    for (auto& iedge : dfg.in_edges(*tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        if (iedge.dst_conn() == "_in1") {
            EXPECT_EQ(iedge.subset().size(), 2);
            ASSERT_GE(iedge.subset().size(), 2);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), i_renamed));
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(1), k_renamed));
            EXPECT_EQ(access_node->data(), "A");
        } else if (iedge.dst_conn() == "_in2") {
            EXPECT_EQ(iedge.subset().size(), 2);
            ASSERT_GE(iedge.subset().size(), 2);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), k_renamed));
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(1), j_renamed));
            EXPECT_EQ(access_node->data(), "B");
        } else if (iedge.dst_conn() == "__einsum_out") {
            EXPECT_EQ(iedge.subset().size(), 2);
            ASSERT_GE(iedge.subset().size(), 2);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), i_renamed));
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(1), j_renamed));
            EXPECT_EQ(access_node->data(), "C");
        } else {
            EXPECT_EQ(iedge.dst_conn(), "Unknown connector");
        }
    }
}

TEST(EinsumNodeTest, SimpleMeans) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add for loop with initialization
    auto& for_init = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));
    auto& block_init = builder.add_block(for_init.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add EinsumNode with summation
    auto& block_sum = builder.add_block(root);
    auto& A_sum = builder.add_access(block_sum, "A");
    auto& y_sum1 = builder.add_access(block_sum, "y");
    auto& y_sum2 = builder.add_access(block_sum, "y");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<
            data_flow::Subset>&>(block_sum, DebugInfo(), {"_in"}, {{i, zero, m}, {j, zero, n}}, {i}, {{i, j}});
    builder.add_computational_memlet(block_sum, A_sum, libnode, "_in", {}, desc_m);
    builder.add_computational_memlet(block_sum, y_sum1, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block_sum, libnode, "__einsum_out", y_sum2, {}, desc);

    // Add for loop with division
    auto& for_div = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));
    auto& block_div = builder.add_block(for_div.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    EXPECT_EQ(einsum_node->dims().size(), 2);
    ASSERT_GE(einsum_node->dims().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(0), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(0), m));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(1), j));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(1), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(1), n));

    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->out_indices().size(), 1);
    ASSERT_GE(einsum_node->out_indices().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(0), i));

    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 2);
    ASSERT_GE(einsum_node->in_indices().size(), 2);
    EXPECT_EQ(einsum_node->in_indices(0).size(), 2);
    ASSERT_GE(einsum_node->in_indices(0).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 1), j));
    EXPECT_EQ(einsum_node->in_indices(1).size(), 1);
    ASSERT_GE(einsum_node->in_indices(1).size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 0), i));

    auto symbols = einsum_node->symbols();
    EXPECT_EQ(symbols.size(), 4);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(n));

    EXPECT_EQ(einsum_node->toStr(), "__einsum_out[i] = _in[i][j] + __einsum_out[i] for i = 0 : m for j = 0 : n");

    EXPECT_TRUE(symbolic::eq(einsum_node->flop(), symbolic::mul(m, n)));

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(EinsumNodeTest, ExpandMeans) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add for loop with initialization
    auto& for_init = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));
    auto& block_init = builder.add_block(for_init.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add EinsumNode with summation
    auto& block_sum = builder.add_block(root);
    auto& A_sum = builder.add_access(block_sum, "A");
    auto& y_sum1 = builder.add_access(block_sum, "y");
    auto& y_sum2 = builder.add_access(block_sum, "y");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<
            data_flow::Subset>&>(block_sum, DebugInfo(), {"_in"}, {{i, zero, m}, {j, zero, n}}, {i}, {{i, j}});
    builder.add_computational_memlet(block_sum, A_sum, libnode, "_in", {}, desc_m);
    builder.add_computational_memlet(block_sum, y_sum1, libnode, "__einsum_out", {}, desc);
    builder.add_computational_memlet(block_sum, libnode, "__einsum_out", y_sum2, {}, desc);

    // Add for loop with division
    auto& for_div = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));
    auto& block_div = builder.add_block(for_div.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(einsum_node->expand(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* for_init_opt = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_TRUE(for_init_opt);
    EXPECT_EQ(for_init_opt, &for_init);

    auto* loop1 = dynamic_cast<structured_control_flow::Map*>(&root.at(1).first);
    ASSERT_TRUE(loop1);
    // Indvars are renamed during expansion for uniqueness
    auto i_renamed = loop1->indvar();
    EXPECT_TRUE(symbolic::eq(loop1->init(), zero));
    EXPECT_TRUE(symbolic::eq(loop1->condition(), symbolic::Lt(i_renamed, m)));
    EXPECT_TRUE(symbolic::eq(loop1->update(), symbolic::add(i_renamed, one)));
    EXPECT_EQ(loop1->root().size(), 1);
    ASSERT_GE(loop1->root().size(), 1);

    auto* loop2 = dynamic_cast<structured_control_flow::For*>(&loop1->root().at(0).first);
    ASSERT_TRUE(loop2);
    auto j_renamed = loop2->indvar();
    EXPECT_TRUE(symbolic::eq(loop2->init(), zero));
    EXPECT_TRUE(symbolic::eq(loop2->condition(), symbolic::Lt(j_renamed, n)));
    EXPECT_TRUE(symbolic::eq(loop2->update(), symbolic::add(j_renamed, one)));
    EXPECT_EQ(loop2->root().size(), 1);
    ASSERT_GE(loop2->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&loop2->root().at(0).first);
    ASSERT_TRUE(new_block);
    auto& dfg = new_block->dataflow();
    EXPECT_EQ(dfg.tasklets().size(), 1);
    auto* tasklet = *dfg.tasklets().begin();
    ASSERT_TRUE(tasklet);
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_add);
    EXPECT_EQ(tasklet->output(), "_out");
    for (auto& oedge : dfg.out_edges(*tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 1);
        ASSERT_GE(oedge.subset().size(), 1);
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), i_renamed));
        EXPECT_EQ(access_node->data(), "y");
    }
    EXPECT_EQ(tasklet->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    for (auto& iedge : dfg.in_edges(*tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        if (iedge.dst_conn() == "_in") {
            EXPECT_EQ(iedge.subset().size(), 2);
            ASSERT_GE(iedge.subset().size(), 2);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), i_renamed));
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(1), j_renamed));
            EXPECT_EQ(access_node->data(), "A");
        } else if (iedge.dst_conn() == "__einsum_out") {
            EXPECT_EQ(iedge.subset().size(), 1);
            ASSERT_GE(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), i_renamed));
            EXPECT_EQ(access_node->data(), "y");
        } else {
            EXPECT_EQ(iedge.dst_conn(), "Unknown connector");
        }
    }

    auto* for_div_opt = dynamic_cast<structured_control_flow::For*>(&root.at(2).first);
    ASSERT_TRUE(for_div_opt);
    EXPECT_EQ(for_div_opt, &for_div);
}

TEST(EinsumNodeTest, SimpleMean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add block with ...
    auto& block = builder.add_block(root);
    auto& zero_init = builder.add_constant(block, "0.0", base_desc);
    auto& y1 = builder.add_access(block, "y");
    auto& a_sum = builder.add_access(block, "a");
    auto& y2 = builder.add_access(block, "y");
    auto& m_div = builder.add_access(block, "m");
    auto& m_tmp = builder.add_access(block, "m_tmp");
    auto& y3 = builder.add_access(block, "y");

    // ... initialization
    auto& tasklet_init = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block, tasklet_init, "_out", y1, {});

    // ... summation
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {{i, zero, m}}, {}, {{i}});
    builder.add_computational_memlet(block, a_sum, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, y1, libnode, "__einsum_out", {}, base_desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", y2, {}, base_desc);

    // ... division
    auto& tasklet_div1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, y2, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block, tasklet_div2, "_out", y3, {});

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    EXPECT_EQ(einsum_node->dims().size(), 1);
    ASSERT_GE(einsum_node->dims().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(0), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(0), m));

    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->out_indices().size(), 0);

    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 2);
    ASSERT_GE(einsum_node->in_indices().size(), 2);
    EXPECT_EQ(einsum_node->in_indices(0).size(), 1);
    ASSERT_GE(einsum_node->in_indices(0).size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), i));
    EXPECT_EQ(einsum_node->in_indices(1).size(), 0);

    auto symbols = einsum_node->symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(m));

    EXPECT_EQ(einsum_node->toStr(), "__einsum_out = _in[i] + __einsum_out for i = 0 : m");

    EXPECT_TRUE(symbolic::eq(einsum_node->flop(), m));

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(EinsumNodeTest, ExpandMean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add block with ...
    auto& block = builder.add_block(root);
    auto& zero_init = builder.add_constant(block, "0.0", base_desc);
    auto& y1 = builder.add_access(block, "y");
    auto& a_sum = builder.add_access(block, "a");
    auto& y2 = builder.add_access(block, "y");
    auto& m_div = builder.add_access(block, "m");
    auto& m_tmp = builder.add_access(block, "m_tmp");
    auto& y3 = builder.add_access(block, "y");

    // ... initialization
    auto& tasklet_init = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block, tasklet_init, "_out", y1, {});

    // ... summation
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(block, DebugInfo(), {"_in"}, {{i, zero, m}}, {}, {{i}});
    builder.add_computational_memlet(block, a_sum, libnode, "_in", {}, desc);
    builder.add_computational_memlet(block, y1, libnode, "__einsum_out", {}, base_desc);
    builder.add_computational_memlet(block, libnode, "__einsum_out", y2, {}, base_desc);

    // ... division
    auto& tasklet_div1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, y2, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block, tasklet_div2, "_out", y3, {});

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(einsum_node->expand(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* block_before = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(block_before);
    auto& dfg_before = block_before->dataflow();
    EXPECT_EQ(dfg_before.tasklets().size(), 1);
    ASSERT_GE(dfg_before.tasklets().size(), 1);
    auto* tasklet_before = *dfg_before.tasklets().begin();
    ASSERT_TRUE(tasklet_before);
    EXPECT_EQ(tasklet_before->code(), data_flow::TaskletCode::assign);
    EXPECT_EQ(tasklet_before->output(), "_out");
    for (auto& oedge : dfg_before.out_edges(*tasklet_before)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "y");
    }
    EXPECT_EQ(tasklet_before->inputs(), std::vector<std::string>({"_in"}));
    for (auto& iedge : dfg_before.in_edges(*tasklet_before)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "0.0");
    }

    auto* for1 = dynamic_cast<structured_control_flow::For*>(&root.at(1).first);
    ASSERT_TRUE(for1);
    // Indvars are renamed during expansion for uniqueness
    auto i_renamed = for1->indvar();
    EXPECT_TRUE(symbolic::eq(for1->init(), zero));
    EXPECT_TRUE(symbolic::eq(for1->condition(), symbolic::Lt(i_renamed, m)));
    EXPECT_TRUE(symbolic::eq(for1->update(), symbolic::add(i_renamed, symbolic::one())));
    EXPECT_EQ(for1->root().size(), 1);
    ASSERT_GE(for1->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for1->root().at(0).first);
    ASSERT_TRUE(new_block);
    auto& new_dfg = new_block->dataflow();
    EXPECT_EQ(new_dfg.tasklets().size(), 1);
    ASSERT_GE(new_dfg.tasklets().size(), 1);
    auto* new_tasklet = *new_dfg.tasklets().begin();
    ASSERT_TRUE(new_tasklet);
    EXPECT_EQ(new_tasklet->code(), data_flow::TaskletCode::fp_add);
    EXPECT_EQ(new_tasklet->output(), "_out");
    for (auto& oedge : new_dfg.out_edges(*new_tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "y");
    }
    EXPECT_EQ(new_tasklet->inputs(), std::vector<std::string>({"_in", "__einsum_out"}));
    for (auto& iedge : new_dfg.in_edges(*new_tasklet)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        if (iedge.dst_conn() == "_in") {
            EXPECT_EQ(iedge.subset().size(), 1);
            ASSERT_GE(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), i_renamed));
            EXPECT_EQ(access_node->data(), "a");
        } else if (iedge.dst_conn() == "__einsum_out") {
            EXPECT_EQ(iedge.subset().size(), 0);
            EXPECT_EQ(access_node->data(), "y");
        } else {
            EXPECT_EQ(iedge.dst_conn(), "Unknown connector");
        }
    }

    auto* block_after = dynamic_cast<structured_control_flow::Block*>(&root.at(2).first);
    ASSERT_TRUE(block_after);
    auto& dfg_after = block_after->dataflow();
    EXPECT_EQ(dfg_after.tasklets().size(), 2);
    ASSERT_GE(dfg_after.tasklets().size(), 2);
    data_flow::Tasklet *tasklet_after_1 = nullptr, *tasklet_after_2 = nullptr;
    {
        auto* tasklet1 = *dfg_after.tasklets().begin();
        auto& oedge1 = *dfg_after.out_edges(*tasklet1).begin();
        auto* access1 = dynamic_cast<data_flow::AccessNode*>(&oedge1.dst());
        if (access1 && access1->data() == "m_tmp") {
            tasklet_after_1 = tasklet1;
        } else if (access1 && access1->data() == "y") {
            tasklet_after_2 = tasklet1;
        }
        auto* tasklet2 = *std::next(dfg_after.tasklets().begin());
        auto& oedge2 = *dfg_after.out_edges(*tasklet2).begin();
        auto* access2 = dynamic_cast<data_flow::AccessNode*>(&oedge2.dst());
        if (access2 && access2->data() == "m_tmp") {
            tasklet_after_1 = tasklet2;
        } else if (access2 && access2->data() == "y") {
            tasklet_after_2 = tasklet2;
        }
    }
    ASSERT_TRUE(tasklet_after_1);
    EXPECT_EQ(tasklet_after_1->code(), data_flow::TaskletCode::assign);
    EXPECT_EQ(tasklet_after_1->output(), "_out");
    for (auto& oedge : dfg_after.out_edges(*tasklet_after_1)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "m_tmp");
    }
    EXPECT_EQ(tasklet_after_1->inputs(), std::vector<std::string>({"_in"}));
    for (auto& iedge : dfg_after.in_edges(*tasklet_after_1)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "m");
    }
    ASSERT_TRUE(tasklet_after_2);
    EXPECT_EQ(tasklet_after_2->code(), data_flow::TaskletCode::fp_div);
    EXPECT_EQ(tasklet_after_2->output(), "_out");
    for (auto& oedge : dfg_after.out_edges(*tasklet_after_2)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        ASSERT_TRUE(access_node);
        EXPECT_EQ(oedge.src_conn(), "_out");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(access_node->data(), "y");
    }
    EXPECT_EQ(tasklet_after_2->inputs(), std::vector<std::string>({"_in1", "_in2"}));
    for (auto& iedge : dfg_after.in_edges(*tasklet_after_2)) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        ASSERT_TRUE(access_node);
        if (iedge.dst_conn() == "_in1") {
            EXPECT_EQ(iedge.subset().size(), 0);
            EXPECT_EQ(access_node->data(), "y");
        } else if (iedge.dst_conn() == "_in2") {
            EXPECT_EQ(iedge.subset().size(), 0);
            EXPECT_EQ(access_node->data(), "m_tmp");
        } else {
            EXPECT_EQ(iedge.dst_conn(), "Unknown connector");
        }
    }
}

TEST(EinsumSerializerTest, Serialize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add block with EinsumNode for GEMM
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block, DebugInfo(), {"_in1", "_in2"}, {{i, zero, l}, {j, zero, m}, {k, zero, n}}, {i, j}, {{i, k}, {k, j}}
    );
    builder.add_computational_memlet(block, A, libnode, "_in1", {}, desc_n);
    builder.add_computational_memlet(block, B, libnode, "_in2", {}, desc_m);
    builder.add_computational_memlet(block, C1, libnode, "__einsum_out", {}, desc_m);
    builder.add_computational_memlet(block, libnode, "__einsum_out", C2, {}, desc_m);

    einsum::EinsumSerializer serializer;
    auto libnode_j = serializer.serialize(libnode);

    EXPECT_TRUE(libnode_j.contains("type"));
    EXPECT_TRUE(libnode_j["type"].is_string());
    EXPECT_EQ(libnode_j["type"].get<std::string>(), "library_node");

    EXPECT_TRUE(libnode_j.contains("code"));
    EXPECT_TRUE(libnode_j["code"].is_string());
    EXPECT_EQ(libnode_j["code"].get<std::string>(), "Einsum");

    EXPECT_TRUE(libnode_j.contains("side_effect"));
    EXPECT_TRUE(libnode_j["side_effect"].is_boolean());
    EXPECT_FALSE(libnode_j["side_effect"].get<bool>());

    EXPECT_TRUE(libnode_j.contains("output"));
    EXPECT_TRUE(libnode_j["output"].is_string());
    EXPECT_EQ(libnode_j["output"].get<std::string>(), "__einsum_out");

    EXPECT_TRUE(libnode_j.contains("inputs"));
    EXPECT_TRUE(libnode_j["inputs"].is_array());
    EXPECT_EQ(
        libnode_j["inputs"].get<std::vector<std::string>>(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"})
    );

    EXPECT_TRUE(libnode_j.contains("dims"));
    EXPECT_TRUE(libnode_j["dims"].is_array());
    EXPECT_EQ(libnode_j["dims"].size(), 3);
    ASSERT_GE(libnode_j["dims"].size(), 3);

    EXPECT_TRUE(libnode_j["dims"][0].is_object());
    EXPECT_TRUE(libnode_j["dims"][0].contains("indvar"));
    EXPECT_TRUE(libnode_j["dims"][0]["indvar"].is_string());
    EXPECT_EQ(libnode_j["dims"][0]["indvar"].get<std::string>(), "i");
    EXPECT_TRUE(libnode_j["dims"][0].contains("init"));
    EXPECT_TRUE(libnode_j["dims"][0]["init"].is_string());
    EXPECT_EQ(libnode_j["dims"][0]["init"].get<std::string>(), "0");
    EXPECT_TRUE(libnode_j["dims"][0].contains("bound"));
    EXPECT_TRUE(libnode_j["dims"][0]["bound"].is_string());
    EXPECT_EQ(libnode_j["dims"][0]["bound"].get<std::string>(), "l");

    EXPECT_TRUE(libnode_j["dims"][1].is_object());
    EXPECT_TRUE(libnode_j["dims"][1].contains("indvar"));
    EXPECT_TRUE(libnode_j["dims"][1]["indvar"].is_string());
    EXPECT_EQ(libnode_j["dims"][1]["indvar"].get<std::string>(), "j");
    EXPECT_TRUE(libnode_j["dims"][1].contains("init"));
    EXPECT_TRUE(libnode_j["dims"][1]["init"].is_string());
    EXPECT_EQ(libnode_j["dims"][1]["init"].get<std::string>(), "0");
    EXPECT_TRUE(libnode_j["dims"][1].contains("bound"));
    EXPECT_TRUE(libnode_j["dims"][1]["bound"].is_string());
    EXPECT_EQ(libnode_j["dims"][1]["bound"].get<std::string>(), "m");

    EXPECT_TRUE(libnode_j["dims"][2].is_object());
    EXPECT_TRUE(libnode_j["dims"][2].contains("indvar"));
    EXPECT_TRUE(libnode_j["dims"][2]["indvar"].is_string());
    EXPECT_EQ(libnode_j["dims"][2]["indvar"].get<std::string>(), "k");
    EXPECT_TRUE(libnode_j["dims"][2].contains("init"));
    EXPECT_TRUE(libnode_j["dims"][2]["init"].is_string());
    EXPECT_EQ(libnode_j["dims"][2]["init"].get<std::string>(), "0");
    EXPECT_TRUE(libnode_j["dims"][2].contains("bound"));
    EXPECT_TRUE(libnode_j["dims"][2]["bound"].is_string());
    EXPECT_EQ(libnode_j["dims"][2]["bound"].get<std::string>(), "n");

    EXPECT_TRUE(libnode_j.contains("out_indices"));
    EXPECT_TRUE(libnode_j["out_indices"].is_array());
    EXPECT_EQ(libnode_j["out_indices"].get<std::vector<std::string>>(), std::vector<std::string>({"i", "j"}));

    EXPECT_TRUE(libnode_j.contains("in_indices"));
    EXPECT_TRUE(libnode_j["in_indices"].is_array());
    EXPECT_EQ(
        libnode_j["in_indices"].get<std::vector<std::vector<std::string>>>(),
        std::vector<std::vector<std::string>>(
            {std::vector<std::string>({"i", "k"}),
             std::vector<std::string>({"k", "j"}),
             std::vector<std::string>({"i", "j"})}
        )
    );
}

TEST(EinsumSerializerTest, Deserialize) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add block with access nodes
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");

    // Deserialize einsum node for GEMM
    einsum::EinsumSerializer serializer;
    const std::string json = R"(
{
    "type": "library_node",
    "code": "Einsum",
    "side_effect": false,
    "output": "__einsum_out",
    "inputs": ["_in1", "_in2", "__einsum_out"],
    "dims": [
        {
            "indvar": "i",
            "init": "0",
            "bound": "l"
        },
        {
            "indvar": "j",
            "init": "0",
            "bound": "m"
        },
        {
            "indvar": "k",
            "init": "0",
            "bound": "n"
        }
    ],
    "out_indices": ["i", "j"],
    "in_indices": [
        ["i", "k"],
        ["k", "j"],
        ["i", "j"]
    ]
}
)";
    nlohmann::json libnode_j = nlohmann::json::parse(json);
    auto& libnode = serializer.deserialize(libnode_j, builder, block);

    // Add memlets
    builder.add_computational_memlet(block, A, libnode, "_in1", {}, desc_n);
    builder.add_computational_memlet(block, B, libnode, "_in2", {}, desc_m);
    builder.add_computational_memlet(block, C1, libnode, "__einsum_out", {}, desc_m);
    builder.add_computational_memlet(block, libnode, "__einsum_out", C2, {}, desc_m);

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    EXPECT_EQ(einsum_node->dims().size(), 3);
    ASSERT_GE(einsum_node->dims().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(0), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(0), l));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(1), j));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(1), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(1), m));
    EXPECT_TRUE(symbolic::eq(einsum_node->indvar(2), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->init(2), zero));
    EXPECT_TRUE(symbolic::eq(einsum_node->bound(2), n));

    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->out_indices().size(), 2);
    ASSERT_GE(einsum_node->out_indices().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->out_index(1), j));

    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
    EXPECT_EQ(einsum_node->in_indices().size(), 3);
    ASSERT_GE(einsum_node->in_indices().size(), 3);
    EXPECT_EQ(einsum_node->in_indices(0).size(), 2);
    ASSERT_GE(einsum_node->in_indices(0).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(0, 1), k));
    EXPECT_EQ(einsum_node->in_indices(1).size(), 2);
    ASSERT_GE(einsum_node->in_indices(1).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 0), k));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(1, 1), j));
    EXPECT_EQ(einsum_node->in_indices(2).size(), 2);
    ASSERT_GE(einsum_node->in_indices(2).size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 0), i));
    EXPECT_TRUE(symbolic::eq(einsum_node->in_index(2, 1), j));

    auto symbols = einsum_node->symbols();
    EXPECT_EQ(symbols.size(), 6);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(k));
    EXPECT_TRUE(symbols.contains(l));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(n));

    EXPECT_EQ(
        einsum_node->toStr(),
        "__einsum_out[i][j] = _in1[i][k] * _in2[k][j] + __einsum_out[i][j] for i = 0 : l for j = 0 : m for k = 0 : n"
    );

    EXPECT_TRUE(symbolic::eq(einsum_node->flop(), SymEngine::mul({l, m, n, symbolic::integer(2)})));

    EXPECT_NO_THROW(sdfg.validate());
}
