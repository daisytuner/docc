#include "sdfg/passes/redundant_load_elimination_pass.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

// A tasklet computes a value, the result is stored into A[0] (an indirect /
// pointed-to write) and then immediately read back from A[0]. The pass must
// reroute the read directly from the producing tasklet (via a scalar bypass)
// while keeping the original write intact.
TEST(RedundantLoadEliminationPassTest, RedundantLoad_ReroutesReadAndKeepsWrite) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Pointer desc_pointer(desc_element);
    builder.add_container("A", desc_pointer);
    builder.add_container("B", desc_pointer);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // tasklet computing A[0] = 1.0 + 2.0
    auto& one_node = builder.add_constant(block, "1", desc_element);
    auto& two_node = builder.add_constant(block, "2", desc_element);
    auto& acc_a = builder.add_access(block, "A");
    auto& tasklet_write = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, one_node, tasklet_write, "_in1", {});
    builder.add_computational_memlet(block, two_node, tasklet_write, "_in2", {});
    builder.add_computational_memlet(block, tasklet_write, "_out", acc_a, {symbolic::integer(0)}, desc_pointer);

    // immediate read of A[0] forwarded into B[0]
    auto& acc_b = builder.add_access(block, "B");
    auto& tasklet_read = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, acc_a, tasklet_read, "_in", {symbolic::integer(0)}, desc_pointer);
    builder.add_computational_memlet(block, tasklet_read, "_out", acc_b, {symbolic::integer(0)}, desc_pointer);

    dump_sdfg(builder.subject(), "0.init");

    auto sdfg = builder.move();
    EXPECT_EQ(sdfg->root().size(), 1);

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantLoadEliminationPass pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    dump_sdfg(*sdfg, "1.rle");
    EXPECT_EQ(sdfg->root().size(), 1);

    // A scalar bypass container must have been introduced.
    bool found_bypass = false;
    for (auto& name : sdfg->containers()) {
        if (name.rfind("rle_", 0) == 0) {
            found_bypass = true;
        }
    }
    EXPECT_TRUE(found_bypass);

    auto* result_block = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0));
    ASSERT_NE(result_block, nullptr);
    auto& dataflow = result_block->dataflow();

    // Original 6 nodes + bypass access + copy tasklet = 8 nodes.
    EXPECT_EQ(dataflow.nodes().size(), 8);
    // Original 5 edges become 7 (write split into producer->bypass->copy->A, plus rerouted read).
    EXPECT_EQ(dataflow.edges().size(), 7);

    // The indirect write into A is preserved: A still receives exactly one write.
    size_t a_writes = 0;
    for (auto& node : dataflow.nodes()) {
        if (auto* an = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (dynamic_cast<const data_flow::ConstantNode*>(an) == nullptr && an->data() == "A") {
                a_writes += dataflow.in_degree(*an);
            }
        }
    }
    EXPECT_EQ(a_writes, 1);
}

// Two writes to the very same index A[0] appear in one block, with a read of
// A[0] in between. Because the graph has no order-dependency edges, the first
// (redundant) write is simply removed since it would be overwritten by the
// second write anyway. The first access node is left without any incident
// edges, the second write is kept.
TEST(RedundantLoadEliminationPassTest, ConsecutiveWriteSameIndex_RemovesFirstWrite) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Pointer desc_pointer(desc_element);
    builder.add_container("A", desc_pointer);
    builder.add_container("s", desc_element);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // First write: A[0] = 1.0
    auto& const0 = builder.add_constant(block, "1", desc_element);
    auto& acc_a1 = builder.add_access(block, "A");
    auto& tasklet_w1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const0, tasklet_w1, "_in", {});
    builder.add_computational_memlet(block, tasklet_w1, "_out", acc_a1, {symbolic::integer(0)}, desc_pointer);

    // Read of A[0] into scalar s (forces an ordering A1 -> ... -> A2 in topo sort).
    auto& acc_s = builder.add_access(block, "s");
    auto& tasklet_read = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, acc_a1, tasklet_read, "_in", {symbolic::integer(0)}, desc_pointer);
    builder.add_computational_memlet(block, tasklet_read, "_out", acc_s, {});

    // Second write to the same index: A[0] = s
    auto& acc_a2 = builder.add_access(block, "A");
    auto& tasklet_w2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, acc_s, tasklet_w2, "_in", {});
    builder.add_computational_memlet(block, tasklet_w2, "_out", acc_a2, {symbolic::integer(0)}, desc_pointer);

    dump_sdfg(builder.subject(), "0.init");

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantLoadEliminationPass pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    dump_sdfg(*sdfg, "1.rle");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto* result_block = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0));
    ASSERT_NE(result_block, nullptr);
    auto& dataflow = result_block->dataflow();

    // Of the two access nodes for "A", exactly one is fully disconnected (the
    // removed redundant write) and exactly one still carries the kept write.
    size_t disconnected_a = 0;
    size_t written_a = 0;
    for (auto& node : dataflow.nodes()) {
        if (auto* an = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (dynamic_cast<const data_flow::ConstantNode*>(an) == nullptr && an->data() == "A") {
                if (dataflow.in_degree(*an) == 0 && dataflow.out_degree(*an) == 0) {
                    disconnected_a++;
                } else if (dataflow.in_degree(*an) == 1) {
                    written_a++;
                }
            }
        }
    }
    EXPECT_EQ(disconnected_a, 1);
    EXPECT_EQ(written_a, 1);

    // No copy-back tasklet is generated for a redundant write, so the bypass is
    // wired directly; the original first write is gone.
    bool found_bypass = false;
    for (auto& name : sdfg->containers()) {
        if (name.rfind("rle_", 0) == 0) {
            found_bypass = true;
        }
    }
    EXPECT_TRUE(found_bypass);
}

// Regression test: an access node with more than one incoming write edge (here
// two writes at different offsets A[0] and A[1]) must neither match the
// redundant-load criteria nor crash the pass. Such an access node only shares
// the base pointer between independent writes at distinct subsets.
TEST(RedundantLoadEliminationPassTest, DoesNotCrashOn_MultiInputAccessNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Pointer desc_pointer(desc_element);
    builder.add_container("A", desc_pointer);
    builder.add_container("B", desc_pointer);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    auto& acc_a = builder.add_access(block, "A");

    // Write A[0] = 1.0
    auto& const0 = builder.add_constant(block, "1", desc_element);
    auto& tasklet_w0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const0, tasklet_w0, "_in", {});
    builder.add_computational_memlet(block, tasklet_w0, "_out", acc_a, {symbolic::integer(0)}, desc_pointer);

    // Write A[1] = 2.0 (different offset, same base pointer)
    auto& const1 = builder.add_constant(block, "2", desc_element);
    auto& tasklet_w1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const1, tasklet_w1, "_in", {});
    builder.add_computational_memlet(block, tasklet_w1, "_out", acc_a, {symbolic::integer(1)}, desc_pointer);

    // Read A[0] into B[0]
    auto& acc_b = builder.add_access(block, "B");
    auto& tasklet_read = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, acc_a, tasklet_read, "_in", {symbolic::integer(0)}, desc_pointer);
    builder.add_computational_memlet(block, tasklet_read, "_out", acc_b, {symbolic::integer(0)}, desc_pointer);

    dump_sdfg(builder.subject(), "0.init");

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantLoadEliminationPass pass;
    // Multiple input edges disqualify the access node: no optimization, no crash.
    EXPECT_FALSE(pass.run_pass(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    dump_sdfg(*sdfg, "1.rle");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto* result_block = dynamic_cast<const structured_control_flow::Block*>(&sdfg->root().at(0));
    ASSERT_NE(result_block, nullptr);
    auto& dataflow = result_block->dataflow();

    // Both writes to A are still present.
    size_t a_writes = 0;
    for (auto& node : dataflow.nodes()) {
        if (auto* an = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (dynamic_cast<const data_flow::ConstantNode*>(an) == nullptr && an->data() == "A") {
                a_writes += dataflow.in_degree(*an);
            }
        }
    }
    EXPECT_EQ(a_writes, 2);
}
