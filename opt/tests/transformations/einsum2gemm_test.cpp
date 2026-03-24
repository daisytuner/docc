#include "sdfg/transformations/einsum2gemm.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
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

TEST(Einsum2GemmTest, Simple) {
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
    auto one = symbolic::one();
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

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::Einsum2Gemm transformation(*einsum_node, "openmp");
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 6);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(*dfg.library_nodes().begin());
    ASSERT_TRUE(gemm_node);
    EXPECT_EQ(gemm_node->layout(), math::blas::BLAS_Layout::RowMajor);
    EXPECT_EQ(gemm_node->trans_a(), math::blas::BLAS_Transpose::No);
    EXPECT_EQ(gemm_node->trans_b(), math::blas::BLAS_Transpose::No);
    EXPECT_TRUE(symbolic::eq(gemm_node->m(), l));
    EXPECT_TRUE(symbolic::eq(gemm_node->n(), m));
    EXPECT_TRUE(symbolic::eq(gemm_node->k(), n));
    EXPECT_TRUE(symbolic::eq(gemm_node->lda(), n));
    EXPECT_TRUE(symbolic::eq(gemm_node->ldb(), m));
    EXPECT_TRUE(symbolic::eq(gemm_node->ldc(), m));

    auto conn2cont = get_conn2cont(dfg, *gemm_node);
    EXPECT_EQ(conn2cont.at("__A"), "A");
    EXPECT_EQ(conn2cont.at("__B"), "B");
    EXPECT_EQ(conn2cont.at("__C"), "C");
    EXPECT_EQ(conn2cont.at("__alpha"), "1.0");
    EXPECT_EQ(conn2cont.at("__beta"), "1.0");
}

TEST(Einsum2GemmTest, WithAlpha) {
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
    builder.add_container("alpha", base_desc, true);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
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
    auto& alpha = builder.add_access(block, "alpha");
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        block,
        DebugInfo(),
        {"_in1", "_in2", "_in3"},
        {{i, zero, l}, {j, zero, m}, {k, zero, n}},
        {i, j},
        {{i, k}, {k, j}, {}}
    );
    builder.add_computational_memlet(block, A, libnode, "_in1", {}, desc_n);
    builder.add_computational_memlet(block, B, libnode, "_in2", {}, desc_m);
    builder.add_computational_memlet(block, alpha, libnode, "_in3", {}, base_desc);
    builder.add_computational_memlet(block, C1, libnode, "__einsum_out", {}, desc_m);
    builder.add_computational_memlet(block, libnode, "__einsum_out", C2, {}, desc_m);

    // Check
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::Einsum2Gemm transformation(*einsum_node, "openmp");
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 6);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    ASSERT_GE(dfg.library_nodes().size(), 1);

    auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(*dfg.library_nodes().begin());
    ASSERT_TRUE(gemm_node);
    EXPECT_EQ(gemm_node->layout(), math::blas::BLAS_Layout::RowMajor);
    EXPECT_EQ(gemm_node->trans_a(), math::blas::BLAS_Transpose::No);
    EXPECT_EQ(gemm_node->trans_b(), math::blas::BLAS_Transpose::No);
    EXPECT_TRUE(symbolic::eq(gemm_node->m(), l));
    EXPECT_TRUE(symbolic::eq(gemm_node->n(), m));
    EXPECT_TRUE(symbolic::eq(gemm_node->k(), n));
    EXPECT_TRUE(symbolic::eq(gemm_node->lda(), n));
    EXPECT_TRUE(symbolic::eq(gemm_node->ldb(), m));
    EXPECT_TRUE(symbolic::eq(gemm_node->ldc(), m));

    auto conn2cont = get_conn2cont(dfg, *gemm_node);
    EXPECT_EQ(conn2cont.at("__A"), "A");
    EXPECT_EQ(conn2cont.at("__B"), "B");
    EXPECT_EQ(conn2cont.at("__C"), "C");
    EXPECT_EQ(conn2cont.at("__alpha"), "alpha");
    EXPECT_EQ(conn2cont.at("__beta"), "1.0");
}
