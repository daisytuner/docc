#include "sdfg/data_flow/data_flow_graph.h"
#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/function.h"
#include "sdfg/types/type.h"

using namespace sdfg;

inline static void
check_topo_sort(builder::StructuredSDFGBuilder& builder, const std::vector<data_flow::DataFlowNode*>& expected) {
    std::vector<size_t> expected_ids(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        expected_ids[i] = expected.at(i)->element_id();
    }

    ASSERT_EQ(builder.subject().root().size(), 1) << "StructuredSDFG must have exactly one node";
    auto* block = dyn_cast<structured_control_flow::Block*>(&builder.subject().root().at(0));
    ASSERT_TRUE(block) << "StructuredSDFG must have exactly one block";

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = block->dataflow().topological_sort();
        order.assign(list.begin(), list.end());

        // Verify no duplicates
        std::unordered_set<const data_flow::DataFlowNode*> unique_nodes(order.begin(), order.end());
        EXPECT_EQ(order.size(), unique_nodes.size()) << "Duplicate node(s) in topological sort";
    });
    ASSERT_EQ(order.size(), expected_ids.size());
    for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i]->element_id(), expected_ids[i]) << "Mismatch at index " << i;
    }

    auto sdfg2 = builder.subject().clone();
    builder::StructuredSDFGBuilder builder2(sdfg2);

    ASSERT_EQ(builder2.subject().root().size(), 1) << "Cloned StructuredSDFG must have exactly one node";
    auto* block2 = dyn_cast<structured_control_flow::Block*>(&builder2.subject().root().at(0));
    ASSERT_TRUE(block2) << "Cloned StructuredSDFG must have exactly one block";

    std::vector<const data_flow::DataFlowNode*> order2;
    EXPECT_NO_THROW({
        auto list = block2->dataflow().topological_sort();
        order2.assign(list.begin(), list.end());

        // Verify no duplicates
        std::unordered_set<const data_flow::DataFlowNode*> unique_nodes(order2.begin(), order2.end());
        EXPECT_EQ(order2.size(), unique_nodes.size()) << "Duplicate node(s) in topological sort (cloned)";
    });
    ASSERT_EQ(order2.size(), expected_ids.size());
    for (size_t i = 0; i < order2.size(); ++i) {
        EXPECT_EQ(order2[i]->element_id(), expected_ids[i]) << "Mismatch at index " << i << " (cloned)";
    }
}

TEST(DataflowTest, TopologicalSort) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("scalar_1", desc);
    builder.add_container("scalar_2", desc);

    auto& access_node_1 = builder.add_access(block, "scalar_1");
    auto& access_node_2 = builder.add_access(block, "scalar_2");
    auto& access_node_3 = builder.add_access(block, "scalar_1");
    auto& tasklet_1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(block, tasklet_1, "_out", access_node_2, {});
    builder.add_computational_memlet(block, access_node_2, tasklet_2, "_in", {});
    builder.add_computational_memlet(block, tasklet_2, "_out", access_node_3, {});

    check_topo_sort(builder, {&access_node_1, &tasklet_1, &access_node_2, &tasklet_2, &access_node_3});
}

TEST(DataflowTest, DeterministicTopologicalSort) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);
    builder.add_container("F", desc);
    builder.add_container("G", desc);
    builder.add_container("H", desc);
    builder.add_container("I", desc);
    builder.add_container("J", desc);
    builder.add_container("K", desc);
    builder.add_container("L", desc);
    builder.add_container("M", desc);
    builder.add_container("N", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& E = builder.add_access(block, "E");
    auto& F = builder.add_access(block, "F");
    auto& G = builder.add_access(block, "G");
    auto& H = builder.add_access(block, "H");
    auto& I = builder.add_access(block, "I");
    auto& J = builder.add_access(block, "J");
    auto& K = builder.add_access(block, "K");
    auto& L = builder.add_access(block, "L");
    auto& M = builder.add_access(block, "M");
    auto& N = builder.add_access(block, "N");

    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, B, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", C, {});

    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, F, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, E, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, tasklet2, "_out", G, {});

    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, C, tasklet3, "_in", {});
    builder.add_computational_memlet(block, tasklet3, "_out", D, {});

    auto& tasklet4 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in2", "_in1"});
    builder.add_computational_memlet(block, J, tasklet4, "_in2", {});
    builder.add_computational_memlet(block, K, tasklet4, "_in1", {});
    builder.add_computational_memlet(block, tasklet4, "_out", L, {});

    auto& tasklet5 = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, D, tasklet5, "_in1", {});
    builder.add_computational_memlet(block, G, tasklet5, "_in2", {});
    builder.add_computational_memlet(block, tasklet5, "_out", H, {});

    auto& tasklet6 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, G, tasklet6, "_in", {});
    builder.add_computational_memlet(block, tasklet6, "_out", I, {});

    auto& tasklet7 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, K, tasklet7, "_in", {});
    builder.add_computational_memlet(block, tasklet7, "_out", M, {});

    auto& tasklet8 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, L, tasklet8, "_in1", {});
    builder.add_computational_memlet(block, M, tasklet8, "_in2", {});
    builder.add_computational_memlet(block, tasklet8, "_out", N, {});

    check_topo_sort(builder, {&E,        &F, &tasklet2, &G, &A,        &B, &tasklet1, &C, &tasklet3, &D, &tasklet5, &H,
                              &tasklet6, &I, &K,        &J, &tasklet4, &L, &tasklet7, &M, &tasklet8, &N});
}

TEST(DataflowTest, DiamondGraph) {
    builder::StructuredSDFGBuilder builder("diamond_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", C, {});

    // B, C -> T3 -> D
    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, B, T3, "_in1", {});
    builder.add_computational_memlet(block, C, T3, "_in2", {});
    builder.add_computational_memlet(block, T3, "_out", D, {});

    check_topo_sort(builder, {&A, &T1, &B, &T2, &C, &T3, &D});
}

TEST(DataflowTest, MultiEdgeGraph) {
    builder::StructuredSDFGBuilder builder("multi_edge_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");

    // A -> T1 (two edges)
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    // Two edges from A to T1
    builder.add_computational_memlet(block, A, T1, "_in1", {});
    builder.add_computational_memlet(block, A, T1, "_in2", {});

    builder.add_computational_memlet(block, T1, "_out", B, {});

    check_topo_sort(builder, {&A, &T1, &B});
}

TEST(DataflowTest, CrossedDependencies) {
    builder::StructuredSDFGBuilder builder("crossed_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("S1", desc);
    builder.add_container("S2", desc);

    types::Function func_desc(desc, false);
    func_desc.add_param(desc);
    builder.add_container("func1", func_desc, false, true);
    builder.add_container("func2", func_desc, false, true);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& S1 = builder.add_access(block, "S1");
    auto& S2 = builder.add_access(block, "S2");

    // CallNode T1 (acts as A source logic)
    // Outputs: _out1 (Primary) -> S1, _out2 (Secondary) -> S2
    auto& T1 = builder.add_library_node<data_flow::CallNode>(
        block, DebugInfo(), "func1", std::vector<std::string>{"_out1", "_out2"}, std::vector<std::string>{"_in"}
    );
    builder.add_computational_memlet(block, A, T1, "_in", {}, desc);
    builder.add_computational_memlet(block, T1, "_out1", S1, {}, desc);
    builder.add_computational_memlet(block, T1, "_out2", S2, {}, desc);

    // CallNode T2 (acts as B source logic)
    // Outputs: _out1 (Primary) -> S2, _out2 (Secondary) -> S1
    auto& T2 = builder.add_library_node<data_flow::CallNode>(
        block, DebugInfo(), "func2", std::vector<std::string>{"_out1", "_out2"}, std::vector<std::string>{"_in"}
    );
    builder.add_computational_memlet(block, B, T2, "_in", {}, desc);
    builder.add_computational_memlet(block, T2, "_out1", S2, {}, desc);
    builder.add_computational_memlet(block, T2, "_out2", S1, {}, desc);

    check_topo_sort(builder, {&A, &T1, &B, &T2, &S2, &S1});
}

TEST(DataflowTest, FanOutGraph) {
    builder::StructuredSDFGBuilder builder("fan_out_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("S1", desc);
    builder.add_container("S2", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& S1 = builder.add_access(block, "S1");
    auto& S2 = builder.add_access(block, "S2");

    // A -> B -> S1
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, B, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", S1, {});

    // A -> C -> S2
    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T3, "_in", {});
    builder.add_computational_memlet(block, T3, "_out", C, {});

    auto& T4 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, C, T4, "_in", {});
    builder.add_computational_memlet(block, T4, "_out", S2, {});

    check_topo_sort(builder, {&A, &T1, &B, &T2, &S1, &T3, &C, &T4, &S2});
}

TEST(DataflowTest, ComplexMultiInput) {
    builder::StructuredSDFGBuilder builder("complex_multi_input", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    // A -> T2 -> B
    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", B, {});

    // B -> T3 -> C
    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, B, T3, "_in", {});
    builder.add_computational_memlet(block, T3, "_out", C, {});

    check_topo_sort(builder, {&A, &T1, &T2, &B, &T3, &C});
}

TEST(DataflowTest, ComplexBranching) {
    builder::StructuredSDFGBuilder builder("complex_branching", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& E = builder.add_access(block, "E");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", C, {});

    // B, C -> T3 -> D
    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, B, T3, "_in1", {});
    builder.add_computational_memlet(block, C, T3, "_in2", {});
    builder.add_computational_memlet(block, T3, "_out", D, {});

    // D -> T4 -> E
    auto& T4 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, D, T4, "_in", {});
    builder.add_computational_memlet(block, T4, "_out", E, {});

    check_topo_sort(builder, {&A, &T1, &B, &T2, &C, &T3, &D, &T4, &E});
}

TEST(DataflowTest, TriangleWithCrossEdge) {
    builder::StructuredSDFGBuilder builder("triangle_cross", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");

    // A -> T1 -> B
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    // A -> T2 -> C
    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", C, {});

    // B -> T3 -> C (Cross edge)
    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, B, T3, "_in", {});
    builder.add_computational_memlet(block, T3, "_out", C, {});

    // B -> T4 -> D
    auto& T4 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, B, T4, "_in", {});
    builder.add_computational_memlet(block, T4, "_out", D, {});

    // C -> T5 -> D
    auto& T5 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, C, T5, "_in", {});
    builder.add_computational_memlet(block, T5, "_out", D, {});

    check_topo_sort(builder, {&A, &T1, &B, &T4, &T2, &T3, &C, &T5, &D});
}

TEST(DataflowTest, LinearChain) {
    builder::StructuredSDFGBuilder builder("linear_chain", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& E = builder.add_access(block, "E");

    // A -> T1 -> B -> T2 -> C -> T3 -> D -> T4 -> E
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, T1, "_in", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, B, T2, "_in", {});
    builder.add_computational_memlet(block, T2, "_out", C, {});

    auto& T3 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, C, T3, "_in", {});
    builder.add_computational_memlet(block, T3, "_out", D, {});

    auto& T4 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, D, T4, "_in", {});
    builder.add_computational_memlet(block, T4, "_out", E, {});

    check_topo_sort(builder, {&A, &T1, &B, &T2, &C, &T3, &D, &T4, &E});
}

TEST(DataflowTest, ButterflyPattern) {
    builder::SDFGBuilder builder("butterfly", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);
    builder.add_container("F", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");
    auto& E = builder.add_access(state, "E");
    auto& F = builder.add_access(state, "F");

    // A -> T1 -> C
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", C, {});

    // A -> T2 -> D
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", D, {});

    // B -> T3 -> C
    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", C, {});

    // B -> T4 -> D
    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", D, {});

    // C -> T5 -> E
    auto& T5 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, C, T5, "_in", {});
    builder.add_computational_memlet(state, T5, "_out", E, {});

    // D -> T6 -> F
    auto& T6 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, D, T6, "_in", {});
    builder.add_computational_memlet(state, T6, "_out", F, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort();
        order.assign(list.begin(), list.end());
    });

    // Verify the order is valid (all dependencies respected)
    std::unordered_map<const data_flow::DataFlowNode*, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    // Check dependencies
    EXPECT_LT(position[&A], position[&T1]);
    EXPECT_LT(position[&A], position[&T2]);
    EXPECT_LT(position[&B], position[&T3]);
    EXPECT_LT(position[&B], position[&T4]);
    EXPECT_LT(position[&T1], position[&C]);
    EXPECT_LT(position[&T3], position[&C]);
    EXPECT_LT(position[&T2], position[&D]);
    EXPECT_LT(position[&T4], position[&D]);
    EXPECT_LT(position[&C], position[&T5]);
    EXPECT_LT(position[&D], position[&T6]);
    EXPECT_LT(position[&T5], position[&E]);
    EXPECT_LT(position[&T6], position[&F]);
}

TEST(DataflowTest, MultipleEdgesSameDirection) {
    builder::StructuredSDFGBuilder builder("multi_edges_same_dir", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");

    // A -> T1 -> B (with 3 edges from A to T1)
    auto& T1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, A, T1, "_in1", {});
    builder.add_computational_memlet(block, A, T1, "_in2", {});
    builder.add_computational_memlet(block, A, T1, "_in3", {});
    builder.add_computational_memlet(block, T1, "_out", B, {});

    // B -> T2 -> C (with 2 edges from B to T2)
    auto& T2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, B, T2, "_in1", {});
    builder.add_computational_memlet(block, B, T2, "_in2", {});
    builder.add_computational_memlet(block, T2, "_out", C, {});

    check_topo_sort(builder, {&A, &T1, &B, &T2, &C});
}

TEST(DataflowTest, StarPattern) {
    builder::SDFGBuilder builder("star_pattern", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("Center", desc);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);

    auto& Center = builder.add_access(state, "Center");
    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& C = builder.add_access(state, "C");
    auto& D = builder.add_access(state, "D");

    // Center fans out to A, B, C, D
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Center, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", A, {});

    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Center, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", B, {});

    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Center, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", C, {});

    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Center, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", D, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort();
        order.assign(list.begin(), list.end());
    });

    // Center should come first
    EXPECT_EQ(order[0], &Center);

    // Verify all dependencies
    std::unordered_map<const data_flow::DataFlowNode*, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    EXPECT_LT(position[&Center], position[&T1]);
    EXPECT_LT(position[&Center], position[&T2]);
    EXPECT_LT(position[&Center], position[&T3]);
    EXPECT_LT(position[&Center], position[&T4]);
    EXPECT_LT(position[&T1], position[&A]);
    EXPECT_LT(position[&T2], position[&B]);
    EXPECT_LT(position[&T3], position[&C]);
    EXPECT_LT(position[&T4], position[&D]);
}

// Test: Multiple paths converging to same node via different edges
TEST(DataflowTest, ConvergingPaths) {
    builder::SDFGBuilder builder("converging", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("M1", desc);
    builder.add_container("M2", desc);
    builder.add_container("Z", desc);

    auto& A = builder.add_access(state, "A");
    auto& B = builder.add_access(state, "B");
    auto& M1 = builder.add_access(state, "M1");
    auto& M2 = builder.add_access(state, "M2");
    auto& Z = builder.add_access(state, "Z");

    // A -> T1 -> M1 -> T3 -> Z
    auto& T1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, A, T1, "_in", {});
    builder.add_computational_memlet(state, T1, "_out", M1, {});

    auto& T3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, M1, T3, "_in", {});
    builder.add_computational_memlet(state, T3, "_out", Z, {});

    // B -> T2 -> M2 -> T4 -> Z (multiple edges to same sink)
    auto& T2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, B, T2, "_in", {});
    builder.add_computational_memlet(state, T2, "_out", M2, {});

    auto& T4 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, M2, T4, "_in", {});
    builder.add_computational_memlet(state, T4, "_out", Z, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort();
        order.assign(list.begin(), list.end());
    });

    // Verify all edges are respected
    std::unordered_map<const data_flow::DataFlowNode*, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    // Path 1: A -> T1 -> M1 -> T3 -> Z
    EXPECT_LT(position[&A], position[&T1]);
    EXPECT_LT(position[&T1], position[&M1]);
    EXPECT_LT(position[&M1], position[&T3]);
    EXPECT_LT(position[&T3], position[&Z]);

    // Path 2: B -> T2 -> M2 -> T4 -> Z
    EXPECT_LT(position[&B], position[&T2]);
    EXPECT_LT(position[&T2], position[&M2]);
    EXPECT_LT(position[&M2], position[&T4]);
    EXPECT_LT(position[&T4], position[&Z]);
}

// Test: Diamond with multiple intermediate nodes
TEST(DataflowTest, ComplexDiamond) {
    builder::SDFGBuilder builder("complex_diamond", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("Source", desc);
    builder.add_container("L1", desc);
    builder.add_container("L2", desc);
    builder.add_container("R1", desc);
    builder.add_container("R2", desc);
    builder.add_container("Sink", desc);

    auto& Source = builder.add_access(state, "Source");
    auto& L1 = builder.add_access(state, "L1");
    auto& L2 = builder.add_access(state, "L2");
    auto& R1 = builder.add_access(state, "R1");
    auto& R2 = builder.add_access(state, "R2");
    auto& Sink = builder.add_access(state, "Sink");

    // Left path: Source -> TL1 -> L1 -> TL2 -> L2 -> TL3 -> Sink
    auto& TL1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Source, TL1, "_in", {});
    builder.add_computational_memlet(state, TL1, "_out", L1, {});

    auto& TL2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, L1, TL2, "_in", {});
    builder.add_computational_memlet(state, TL2, "_out", L2, {});

    auto& TL3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, L2, TL3, "_in", {});
    builder.add_computational_memlet(state, TL3, "_out", Sink, {});

    // Right path: Source -> TR1 -> R1 -> TR2 -> R2 -> TR3 -> Sink
    auto& TR1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, Source, TR1, "_in", {});
    builder.add_computational_memlet(state, TR1, "_out", R1, {});

    auto& TR2 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, R1, TR2, "_in", {});
    builder.add_computational_memlet(state, TR2, "_out", R2, {});

    auto& TR3 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, R2, TR3, "_in", {});
    builder.add_computational_memlet(state, TR3, "_out", Sink, {});

    std::vector<const data_flow::DataFlowNode*> order;
    EXPECT_NO_THROW({
        auto list = state.dataflow().topological_sort();
        order.assign(list.begin(), list.end());
    });

    // Verify all edges are respected
    std::unordered_map<const data_flow::DataFlowNode*, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) {
        position[order[i]] = i;
    }

    // Source must be first
    EXPECT_EQ(order[0], &Source);

    // Left path dependencies
    EXPECT_LT(position[&Source], position[&TL1]);
    EXPECT_LT(position[&TL1], position[&L1]);
    EXPECT_LT(position[&L1], position[&TL2]);
    EXPECT_LT(position[&TL2], position[&L2]);
    EXPECT_LT(position[&L2], position[&TL3]);
    EXPECT_LT(position[&TL3], position[&Sink]);

    // Right path dependencies
    EXPECT_LT(position[&Source], position[&TR1]);
    EXPECT_LT(position[&TR1], position[&R1]);
    EXPECT_LT(position[&R1], position[&TR2]);
    EXPECT_LT(position[&TR2], position[&R2]);
    EXPECT_LT(position[&R2], position[&TR3]);
    EXPECT_LT(position[&TR3], position[&Sink]);
}

TEST(DataflowTest, ComplexDependencyResolve) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);
    builder.add_container("D", desc);
    builder.add_container("E", desc);
    builder.add_container("F", desc);
    builder.add_container("G", desc);

    types::Function func_desc(desc);
    func_desc.add_param(desc);
    func_desc.add_param(desc);
    func_desc.add_param(desc);
    func_desc.add_param(desc);
    func_desc.add_param(desc);
    func_desc.add_param(desc);
    builder.add_container("func", func_desc, false, true);

    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& E = builder.add_access(block, "E");
    auto& F = builder.add_access(block, "F");
    auto& G = builder.add_access(block, "G");

    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, B, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", C, {});

    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, C, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, E, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", D, {});

    auto& libnode3 = builder.add_library_node<data_flow::CallNode>(
        block,
        DebugInfo(),
        "func",
        std::vector<std::string>{"_out"},
        std::vector<std::string>{"_in1", "_in2", "_in3", "_in4", "_in5", "_in6"}
    );
    builder.add_computational_memlet(block, D, libnode3, "_in1", {}, desc);
    builder.add_computational_memlet(block, C, libnode3, "_in2", {}, desc);
    builder.add_computational_memlet(block, F, libnode3, "_in3", {}, desc);
    builder.add_computational_memlet(block, A, libnode3, "_in4", {}, desc);
    builder.add_computational_memlet(block, E, libnode3, "_in5", {}, desc);
    builder.add_computational_memlet(block, B, libnode3, "_in6", {}, desc);
    builder.add_computational_memlet(block, libnode3, "_out", G, {}, desc);

    check_topo_sort(builder, {&A, &B, &tasklet1, &C, &E, &tasklet2, &D, &F, &libnode3, &G});
}
