#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_indvar_finalize.h"
#include "sdfg/transformations/loop_rotate.h"
#include "sdfg/transformations/loop_shift.h"
#include "sdfg/transformations/loop_unit_stride.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// Test: Finalize after LoopShift
// for(i=5; i<10; i++) -> normalized to for(i=0; i<5; i++)
// Closed-form: i = num_iterations = 5
// SymbolPropagation will then propagate into reconstruction i = i + 5 -> i = 10
TEST(LoopIndvarFinalizeTest, AfterLoopShift) {
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
    auto* loop = dynamic_cast<structured_control_flow::For*>(&sdfg_root.at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopShift
    transformations::LoopShift shift(*loop);
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    // After shift: loop + 1 reconstruction block
    ASSERT_EQ(sdfg_root.size(), 2);

    // Verify loop is now in normal form
    EXPECT_TRUE(loop->is_loop_normal_form());

    // Apply LoopIndvarFinalize
    transformations::LoopIndvarFinalize finalize(*loop);
    ASSERT_TRUE(finalize.can_be_applied(builder2, am));
    finalize.apply(builder2, am);

    // After finalize: loop + closed-form block + reconstruction block = 3
    ASSERT_EQ(sdfg_root.size(), 3);

    // Check the closed-form block (index 1, right after loop): i = num_iterations = 5
    auto& closed_form_transition = sdfg_root.at(1).second;
    ASSERT_EQ(closed_form_transition.assignments().size(), 1);

    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(closed_form_transition.assignments().count(indvar) > 0);
    auto closed_form = closed_form_transition.assignments().at(indvar);
    auto expected = symbolic::integer(5); // num_iterations = 10 - 5 = 5
    EXPECT_TRUE(symbolic::eq(closed_form, expected));
}

// Test: Finalize after LoopShift + LoopUnitStride
// for(i=2; i<10; i+=2) -> normalized to for(i=0; i<4; i++)
// Closed-form: i = num_iterations = 4
TEST(LoopIndvarFinalizeTest, AfterLoopShiftAndUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 2; i < 10; i += 2 (iterations: 2, 4, 6, 8; exits with i=10)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(2),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
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
    auto* loop = dynamic_cast<structured_control_flow::For*>(&sdfg_root.at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopShift (shifts init from 2 to 0)
    transformations::LoopShift shift(*loop);
    ASSERT_TRUE(shift.can_be_applied(builder2, am));
    shift.apply(builder2, am);

    // After shift: loop + 1 reconstruction block
    ASSERT_EQ(sdfg_root.size(), 2);

    // Apply LoopUnitStride (stride from 2 to 1)
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // After unit stride: loop + 2 reconstruction blocks
    ASSERT_EQ(sdfg_root.size(), 3);

    // Verify loop is now in normal form
    EXPECT_TRUE(loop->is_loop_normal_form());

    // Apply LoopIndvarFinalize
    transformations::LoopIndvarFinalize finalize(*loop);
    ASSERT_TRUE(finalize.can_be_applied(builder2, am));
    finalize.apply(builder2, am);

    // After finalize: loop + closed-form + 2 reconstruction blocks = 4
    ASSERT_EQ(sdfg_root.size(), 4);

    // Check the closed-form block (index 1, right after loop): i = num_iterations = 4
    auto& closed_form_transition = sdfg_root.at(1).second;
    ASSERT_EQ(closed_form_transition.assignments().size(), 1);

    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(closed_form_transition.assignments().count(indvar) > 0);
    auto closed_form = closed_form_transition.assignments().at(indvar);
    auto expected = symbolic::integer(4); // num_iterations = (10 - 2) / 2 = 4
    EXPECT_TRUE(symbolic::eq(closed_form, expected));
}

// Test: Can apply to already normalized loop (adds closed-form)
TEST(LoopIndvarFinalizeTest, CanApplyToNormalizedLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // Already normalized loop: for i = 0; i < 10; i++
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

    auto& sdfg_root = builder2.subject().root();
    auto* loop = dynamic_cast<structured_control_flow::For*>(&sdfg_root.at(0).first);
    ASSERT_NE(loop, nullptr);

    // Loop is already normalized
    EXPECT_TRUE(loop->is_loop_normal_form());

    // Can apply - just adds closed-form
    transformations::LoopIndvarFinalize finalize(*loop);
    EXPECT_TRUE(finalize.can_be_applied(builder2, am));
    finalize.apply(builder2, am);

    // After: loop + closed-form block = 2
    ASSERT_EQ(sdfg_root.size(), 2);

    // Check the closed-form: i = num_iterations = 10
    auto& closed_form_transition = sdfg_root.at(1).second;
    ASSERT_EQ(closed_form_transition.assignments().size(), 1);

    auto indvar = symbolic::symbol("i");
    auto closed_form = closed_form_transition.assignments().at(indvar);
    auto expected = symbolic::integer(10);
    EXPECT_TRUE(symbolic::eq(closed_form, expected));
}

// Test: Cannot apply when loop is not in normal form
TEST(LoopIndvarFinalizeTest, CannotApplyNotNormalForm) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // Non-normalized loop: for i = 5; i < 10; i++
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Loop is not in normal form
    EXPECT_FALSE(loop->is_loop_normal_form());

    transformations::LoopIndvarFinalize finalize(*loop);
    EXPECT_FALSE(finalize.can_be_applied(builder2, am));
}
