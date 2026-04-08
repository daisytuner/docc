#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/loop_shift.h"
#include "sdfg/types/array.h"

using namespace sdfg;

/**
 * Test LoopShift transformation
 *
 * The transformation shifts a loop's iteration space by an offset:
 * - Default constructor: shifts by init value (so loop starts at 0)
 * - Custom offset: shifts by the specified amount
 *
 * After LoopShift:
 * 1. Loop init is reduced by offset
 * 2. A new container __i_orig__ holds the original iteration value
 * 3. All uses of indvar in loop body are replaced with __i_orig__
 * 4. An assignment __i_orig__ = i + offset is added at the start of the body
 */

// Test 1: ShiftToZero - default constructor shifts loop to start at 0
TEST(LoopShiftTest, ShiftToZero) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 3..N
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(3),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Verify initial state: i = 3..N
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(3)));

    // Apply LoopShift with default constructor (shifts to 0)
    transformations::LoopShift shift(*loop);
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    // Verify loop now starts at 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(loop->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::
                    eq(loop->condition(),
                       symbolic::Lt(symbolic::add(symbolic::symbol("i"), symbolic::integer(3)), symbolic::symbol("N")))
    );
    EXPECT_TRUE(symbolic::eq(loop->update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));

    // Verify the shifted container was created
    EXPECT_TRUE(builder2.subject().exists(shift.shifted_container_name()));
    EXPECT_EQ(shift.shifted_container_name(), "__i_orig__");

    // Verify a block was added with the assignment
    ASSERT_EQ(loop->root().size(), 2); // new empty block + original block

    // The first block should have the assignment in its transition
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    ASSERT_EQ(transition.assignments().size(), 1);

    // Assignment should be: __i_orig__ = i + 3
    auto shifted_var = symbolic::symbol(shift.shifted_container_name());
    ASSERT_TRUE(transition.assignments().count(shifted_var) > 0);
    auto assigned_value = transition.assignments().at(shifted_var);
    auto expected = symbolic::add(symbolic::symbol("i"), symbolic::integer(3));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 2: ShiftByOffset - custom offset of 2
TEST(LoopShiftTest, ShiftByOffset) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 5..N
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Verify initial state: i = 5..N
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(5)));

    // Apply LoopShift with offset = 2 (new init = 5 - 2 = 3)
    transformations::LoopShift shift(*loop, symbolic::integer(2));
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    // Verify loop now starts at 3
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(3)));

    // Verify the shifted container was created
    EXPECT_TRUE(builder2.subject().exists(shift.shifted_container_name()));

    // The first block should have the assignment in its transition
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    ASSERT_EQ(transition.assignments().size(), 1);

    // Assignment should be: __i_orig__ = i + 2
    auto shifted_var = symbolic::symbol(shift.shifted_container_name());
    ASSERT_TRUE(transition.assignments().count(shifted_var) > 0);
    auto assigned_value = transition.assignments().at(shifted_var);
    auto expected = symbolic::add(symbolic::symbol("i"), symbolic::integer(2));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 3: IndvarUsedInAccessNode - verify access nodes are updated
TEST(LoopShiftTest, IndvarUsedInAccessNode) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 1..N: A[i] = (double)i  (write indvar value to array)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& i_node = builder.add_access(block, "i");
    auto& a_node = builder.add_access(block, "A");
    // Use fp_cast to convert i to double
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    // Create i as a scalar access (even though it's not in an array, the memlet uses i as the value)
    builder.add_computational_memlet(block, i_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopShift with default constructor (shifts to 0)
    transformations::LoopShift shift(*loop);
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    auto shifted_var = symbolic::symbol(shift.shifted_container_name());

    // Verify loop now starts at 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));

    // Get the second block (original block, now with updated references)
    ASSERT_EQ(loop->root().size(), 2);
    auto* original_block = dynamic_cast<structured_control_flow::Block*>(&loop->root().at(1).first);
    ASSERT_NE(original_block, nullptr);

    // Find the access node for A
    data_flow::AccessNode* indvar_access = nullptr;
    for (auto& node : original_block->dataflow().nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access && access->data() == shifted_var->get_name()) {
            indvar_access = access;
            break;
        }
    }
    ASSERT_NE(indvar_access, nullptr);

    // Check that the memlet subset now uses __i_orig__ instead of i
    auto& dataflow = original_block->dataflow();

    // Find incoming edge to A access node
    for (auto& edge : dataflow.edges()) {
        auto& subset = edge.subset();
        if (subset.empty()) continue; // skip non-memlet edges
        ASSERT_EQ(subset.size(), 1);
        // The subset should now use __i_orig__ instead of i
        EXPECT_TRUE(symbolic::uses(subset[0], shift.shifted_container_name()));
        EXPECT_FALSE(symbolic::uses(subset[0], "i"));
    }
}

// Test 4: NoOpWhenOffsetIsZero - applying shift with offset=0 should be a no-op
TEST(LoopShiftTest, NoOpWhenOffsetIsZero) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 0..N (already starts at 0)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    size_t original_body_size = loop->root().size();

    // Apply LoopShift with default constructor (offset = init = 0)
    transformations::LoopShift shift(*loop);
    ASSERT_FALSE(shift.can_be_applied(builder2, am)); // Should be a no-op
}

// Test: IndvarFinalValueAfterLoop - verify reconstruction after loop
// Original: for(i=5; i<10; i++) body(); // exits with i=10
// After shift: for(i=0; i+5<10; i++) body();  // exits with i=5
// Reconstruction: i = i + 5  // restores i=10 after loop
TEST(LoopShiftTest, IndvarFinalValueAfterLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 5; i < 10; i++
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto& sdfg_root = builder2.subject().root();
    ASSERT_EQ(sdfg_root.size(), 1); // Just the loop

    auto* loop = dynamic_cast<structured_control_flow::For*>(&sdfg_root.at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopShift (shifts to 0 by default, offset = 5)
    transformations::LoopShift shift(*loop);
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    // After transformation, root should have: loop + reconstruction block
    ASSERT_EQ(sdfg_root.size(), 2);

    // Verify the second element is a block (reconstruction)
    auto* post_loop_block = dynamic_cast<structured_control_flow::Block*>(&sdfg_root.at(1).first);
    ASSERT_NE(post_loop_block, nullptr);

    // Check the reconstruction block's transition contains the reconstruction assignment
    // (add_block_after puts assignments in the new block's transition)
    auto& post_loop_transition = sdfg_root.at(1).second;
    ASSERT_EQ(post_loop_transition.assignments().size(), 1);

    // Assignment should be: i = i + 5
    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(post_loop_transition.assignments().count(indvar) > 0);
    auto reconstruction = post_loop_transition.assignments().at(indvar);
    auto expected = symbolic::add(indvar, symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(reconstruction, expected));
}
