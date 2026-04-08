#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/loop_rotate.h"
#include "sdfg/types/array.h"

using namespace sdfg;

/**
 * Test LoopRotate transformation
 *
 * The transformation converts a loop with negative stride to positive stride:
 *
 * Original: for (i = init; bound < i; i--)  body(i)
 * After:    for (i = bound+1; i < init+1; i++)  body(init + bound + 1 - i)
 *
 * Example:
 *   for (i = 10; 0 < i; i--)  =>  for (i = 1; i < 11; i++)
 *   body(i)                       body(11 - i)
 *
 *   Original values: i = 10, 9, 8, ..., 1
 *   New values:      i' = 1, 2, 3, ..., 10  with original_i = 11 - i'
 */

// Test 1: Basic rotation - i = 10..1 (stride = -1, condition: 0 < i)
TEST(LoopRotateTest, BasicRotation) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i-- (i.e., i = 10, 9, ..., 1)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")), // 0 < i
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1)) // i = i - 1
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

    // Verify initial state: stride = -1, init = 10
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), -1);
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(10)));

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);

    // Verify loop now has positive stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // New init = 0 + 1 = 1 (lower bound was 0, exclusive)
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(1)));

    // New condition: i < 10 + 1 = i < 11
    EXPECT_TRUE(symbolic::eq(loop->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(11))));

    // New update: i + 1
    EXPECT_TRUE(symbolic::eq(loop->update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));

    // Verify the rotated container was created
    EXPECT_TRUE(builder2.subject().exists(rotate.rotated_container_name()));
    EXPECT_EQ(rotate.rotated_container_name(), "__i_orig__");

    // Verify a block was added with the assignment
    ASSERT_EQ(loop->root().size(), 2); // new empty block + original block

    // The first block should have the assignment in its transition
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    ASSERT_EQ(transition.assignments().size(), 1);

    // Assignment should be: __i_orig__ = 10 + 1 - i = 11 - i
    auto rotated_var = symbolic::symbol(rotate.rotated_container_name());
    ASSERT_TRUE(transition.assignments().count(rotated_var) > 0);
    auto assigned_value = transition.assignments().at(rotated_var);
    auto expected = symbolic::sub(symbolic::integer(11), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 2: Symbolic bounds - i = N..1 (stride = -1, condition: 0 < i)
TEST(LoopRotateTest, SymbolicBounds) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = N; 0 < i; i--
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::symbol("N"),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
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

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);

    // Verify positive stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // New init = 1
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(1)));

    // New condition: i < N + 1
    EXPECT_TRUE(symbolic::
                    eq(loop->condition(),
                       symbolic::Lt(symbolic::symbol("i"), symbolic::add(symbolic::symbol("N"), symbolic::integer(1))))
    );

    // Assignment should be: __i_orig__ = N + 1 - i
    auto rotated_var = symbolic::symbol(rotate.rotated_container_name());
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto assigned_value = transition.assignments().at(rotated_var);
    auto expected = symbolic::sub(symbolic::add(symbolic::symbol("N"), symbolic::integer(1)), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 3: Lower bound not zero - i = 10..5 (stride = -1, condition: 4 < i)
TEST(LoopRotateTest, NonZeroLowerBound) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 4 < i; i-- (i.e., i = 10, 9, 8, 7, 6, 5)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(4), symbolic::symbol("i")), // 4 < i
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
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

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);

    // New init = 4 + 1 = 5
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(5)));

    // New condition: i < 10 + 1 = i < 11
    EXPECT_TRUE(symbolic::eq(loop->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(11))));

    // Assignment: __i_orig__ = 10 + 5 - i = 15 - i
    // Verify: i' = 5 -> orig = 15 - 5 = 10 ✓
    //         i' = 6 -> orig = 15 - 6 = 9 ✓
    //         ...
    //         i' = 10 -> orig = 15 - 10 = 5 ✓
    auto rotated_var = symbolic::symbol(rotate.rotated_container_name());
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto assigned_value = transition.assignments().at(rotated_var);
    auto expected = symbolic::sub(symbolic::integer(15), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 4: Cannot apply to positive stride
TEST(LoopRotateTest, CannotApplyPositiveStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 0; i < 10; i++ (positive stride)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // LoopRotate should not be applicable
    transformations::LoopRotate rotate(*loop);
    EXPECT_FALSE(rotate.can_be_applied(builder2, am));
}

// Test 5: Cannot apply to stride != -1
TEST(LoopRotateTest, CannotApplyNonUnitNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i -= 2 (stride = -2)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(2)) // stride = -2
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // LoopRotate should not be applicable (stride != -1)
    transformations::LoopRotate rotate(*loop);
    EXPECT_FALSE(rotate.can_be_applied(builder2, am));
}

// Test 6: Work with Map loop
TEST(LoopRotateTest, MapLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // map i = 10..1 (stride = -1)
    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::Map*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);

    // Verify positive stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // Verify it's still a Map
    EXPECT_NE(dynamic_cast<structured_control_flow::Map*>(loop), nullptr);
}

// Test 7: Symbol propagation after rotation
TEST(LoopRotateTest, WithSymbolPropagation) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i--
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
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

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);
    am.invalidate_all();

    // Run SymbolPropagation
    passes::SymbolPropagation symbol_prop;
    bool propagated = symbol_prop.run(builder2, am);
    EXPECT_TRUE(propagated);
    am.invalidate_all();

    // Run DeadDataElimination to clean up
    passes::DeadDataElimination dead_data;
    dead_data.run(builder2, am);
    am.invalidate_all();

    // The __i_orig__ container should have been eliminated after propagation
    // (if DeadDataElimination removes it)
    // Note: this depends on the implementation

    // Run DCE and SequenceFusion to clean up
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder2, am);
        applies |= sequence_fusion.run(builder2, am);
        am.invalidate_all();
    } while (applies);

    // Verify the loop still has positive stride after all cleanup
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);
}

// Test 8: JSON serialization round-trip
TEST(LoopRotateTest, JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Create transformation and serialize to JSON
    transformations::LoopRotate rotate(*loop);
    nlohmann::json j;
    rotate.to_json(j);

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "LoopRotate");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_TRUE(j["subgraph"]["0"].contains("element_id"));

    // Deserialize and verify
    auto rotate2 = transformations::LoopRotate::from_json(builder2, j);
    EXPECT_EQ(rotate2.name(), "LoopRotate");
}

// Test: IndvarFinalValueAfterLoop - verify reconstruction after loop
// Original: for(i=10; 0<i; i--) body(); // exits with i=0
// After rotate: for(i=1; i<11; i++) body();  // exits with i=11
// Reconstruction: i = 10 + 1 - i = 11 - i  // restores i=0 after loop
TEST(LoopRotateTest, IndvarFinalValueAfterLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i-- (exits with i=0)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
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

    // Apply LoopRotate
    transformations::LoopRotate rotate(*loop);
    ASSERT_TRUE(rotate.can_be_applied(builder2, am));
    rotate.apply(builder2, am);

    // After transformation, root should have: loop + reconstruction block
    ASSERT_EQ(sdfg_root.size(), 2);

    // Verify the second element is a block (reconstruction)
    auto* post_loop_block = dynamic_cast<structured_control_flow::Block*>(&sdfg_root.at(1).first);
    ASSERT_NE(post_loop_block, nullptr);

    // Check the reconstruction block's transition contains the reconstruction assignment
    // (add_block_after puts assignments in the new block's transition)
    auto& post_loop_transition = sdfg_root.at(1).second;
    ASSERT_EQ(post_loop_transition.assignments().size(), 1);

    // Assignment should be: i = 10 + 1 - i = 11 - i
    // (old_init=10, new_init=0+1=1, so reconstruction is 10+1-i)
    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(post_loop_transition.assignments().count(indvar) > 0);
    auto reconstruction = post_loop_transition.assignments().at(indvar);
    auto expected = symbolic::sub(symbolic::integer(11), indvar);
    EXPECT_TRUE(symbolic::eq(reconstruction, expected));
}
