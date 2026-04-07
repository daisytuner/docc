#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/transformations/loop_condition_normalize.h"
#include "sdfg/types/array.h"

using namespace sdfg;

/**
 * Test LoopConditionNormalize transformation
 *
 * This transformation converts `!=` (Unequality) conditions to relational
 * comparisons (`<` or `>`) based on the loop's stride direction.
 *
 * For positive stride (+1): i != N -> i < N
 * For negative stride (-1): i != N -> i > N
 */

// Test 1: Basic != to < conversion with positive unit stride
TEST(LoopConditionNormalizeTest, PositiveStrideUnequality) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i != N; i++ (LLVM-style loop condition)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
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

    // Verify initial state: i != N with stride +1
    EXPECT_TRUE(symbolic::eq(loop->stride(), symbolic::integer(1)));
    EXPECT_TRUE(SymEngine::is_a<SymEngine::Unequality>(*loop->condition()));

    // Apply transformation
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // Verify condition was converted to i < N
    auto new_cond = loop->condition();
    // The condition may be (true) && (i < N), so we need to handle CNF
    auto cnf = symbolic::conjunctive_normal_form(new_cond);

    bool found_lt = false;
    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), symbolic::symbol("i")) &&
                    symbolic::eq(lt->get_arg2(), symbolic::symbol("N"))) {
                    found_lt = true;
                }
            }
        }
    }
    EXPECT_TRUE(found_lt) << "Expected i < N in condition, got: " << new_cond->__str__();
}

// Test 2: Negative stride converts != to >
TEST(LoopConditionNormalizeTest, NegativeStrideUnequality) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = N; i != 0; i-- (count down loop)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("i"), symbolic::integer(0)),
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

    // Verify initial state: i != 0 with stride -1
    EXPECT_TRUE(symbolic::eq(loop->stride(), symbolic::integer(-1)));
    EXPECT_TRUE(SymEngine::is_a<SymEngine::Unequality>(*loop->condition()));

    // Apply transformation
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // Verify condition was converted to i > 0
    auto new_cond = loop->condition();
    auto cnf = symbolic::conjunctive_normal_form(new_cond);

    bool found_gt = false;
    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                // i > 0 is represented as 0 < i
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), symbolic::integer(0)) &&
                    symbolic::eq(lt->get_arg2(), symbolic::symbol("i"))) {
                    found_gt = true;
                }
            }
        }
    }
    EXPECT_TRUE(found_gt) << "Expected 0 < i (i > 0) in condition, got: " << new_cond->__str__();
}

// Test 3: Affine expression in condition (i != 2*N + 1)
TEST(LoopConditionNormalizeTest, AffineCondition) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // Affine bound: i != 2*N + 1
    auto bound = symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::symbol("N")), symbolic::integer(1));

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("i"), bound),
        symbolic::integer(0),
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

    // Apply transformation
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // Verify condition was converted to i < 2*N + 1
    auto new_cond = loop->condition();
    auto cnf = symbolic::conjunctive_normal_form(new_cond);

    bool found_lt = false;
    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), symbolic::symbol("i"))) {
                    found_lt = true;
                }
            }
        }
    }
    EXPECT_TRUE(found_lt) << "Expected i < <bound> in condition, got: " << new_cond->__str__();
}

// Test 4: Non-unit stride should not be applicable
TEST(LoopConditionNormalizeTest, NonUnitStride_NotApplicable) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 0; i != N; i += 2 (non-unit stride)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Transformation should not be applicable with stride != ±1
    transformations::LoopConditionNormalize normalize(*loop);
    EXPECT_FALSE(normalize.can_be_applied(builder2, am));
}

// Test 5: Condition without unequality should not be applicable
TEST(LoopConditionNormalizeTest, NoUnequality_NotApplicable) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 0; i < N; i++ (already has < condition)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Transformation should not be applicable (no != to convert)
    transformations::LoopConditionNormalize normalize(*loop);
    EXPECT_FALSE(normalize.can_be_applied(builder2, am));
}

// Test 6: Map with != condition
TEST(LoopConditionNormalizeTest, MapUnequality) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // map i = 0; i != N; i++
    auto& map_loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map_loop.root());
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

    // Apply transformation
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // Verify condition was converted
    auto new_cond = loop->condition();
    auto cnf = symbolic::conjunctive_normal_form(new_cond);

    bool found_lt = false;
    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                found_lt = true;
            }
        }
    }
    EXPECT_TRUE(found_lt) << "Expected < in condition, got: " << new_cond->__str__();
}

// Test 7: Indvar on RHS of condition (N != i)
TEST(LoopConditionNormalizeTest, IndvarOnRHS) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // for i = 0; N != i; i++ (indvar on RHS)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(symbolic::symbol("N"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply transformation
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // Verify condition was converted (should handle indvar on RHS)
    auto new_cond = loop->condition();
    auto cnf = symbolic::conjunctive_normal_form(new_cond);

    bool found_lt = false;
    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                found_lt = true;
            }
        }
    }
    EXPECT_TRUE(found_lt) << "Expected < in condition, got: " << new_cond->__str__();
}

// Test 8: Indvar in non-trivial affine expression: 2*i + 1 != N
// We cannot safely convert this because we don't know if N will be hit
// (depends on divisibility - only works if N is odd)
TEST(LoopConditionNormalizeTest, AffineIndvarExpression_NotApplicable) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // Condition: 2*i + 1 != N (indvar in affine expression with coefficient != 1)
    auto lhs = symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::symbol("i")), symbolic::integer(1));

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Ne(lhs, symbolic::symbol("N")),
        symbolic::integer(0),
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

    // Transformation should NOT be applicable - coefficient of indvar is 2, not 1
    transformations::LoopConditionNormalize normalize(*loop);
    EXPECT_FALSE(normalize.can_be_applied(builder2, am));
}

// Test 9: Boolean comparison pattern - false == (relational)
// This tests the pre-normalization step
TEST(LoopConditionNormalizeTest, BooleanComparisonFalseEqRelational) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // Condition: false == (N < i + M)
    // This means: NOT(N < i + M) = (i + M <= N) = (i <= N - M)
    auto inner_cond = symbolic::Lt(symbolic::symbol("N"), symbolic::add(symbolic::symbol("i"), symbolic::symbol("M")));
    auto cond = symbolic::Eq(symbolic::__false__(), inner_cond);

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        cond,
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Transformation should be applicable (has boolean comparison pattern)
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // After normalization: condition should be simplified
    // false == (N < i + M) → NOT(N < i + M) → (N >= i + M) or equivalently (i + M <= N)
    auto new_cond = loop->condition();

    // Check that it's no longer an Equality with false
    bool is_eq_with_false = false;
    if (SymEngine::is_a<SymEngine::Equality>(*new_cond)) {
        auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(new_cond);
        is_eq_with_false = symbolic::is_false(eq->get_arg1()) || symbolic::is_false(eq->get_arg2());
    }
    EXPECT_FALSE(is_eq_with_false) << "Condition should no longer be (false == ...), got: " << new_cond->__str__();
}

// Test 10: Max pattern - i < max(0, N) with init=0 should become i < N
TEST(LoopConditionNormalizeTest, MaxBoundSimplification) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    // Condition: i < max(0, N)
    auto bound = symbolic::max(symbolic::integer(0), symbolic::symbol("N"));
    auto cond = symbolic::Lt(symbolic::symbol("i"), bound);

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        cond,
        symbolic::integer(0), // init = 0, same as first arg of max
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    builder.add_block(for_loop.root());

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Transformation should be applicable (has max pattern)
    transformations::LoopConditionNormalize normalize(*loop);
    ASSERT_TRUE(normalize.can_be_applied(builder2, am));
    normalize.apply(builder2, am);

    // After normalization: i < max(0, N) should become i < N
    auto new_cond = loop->condition();

    // Check that the bound no longer contains Max
    bool has_max = false;
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*new_cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(new_cond);
        has_max = SymEngine::is_a<SymEngine::Max>(*lt->get_arg2());
    }
    EXPECT_FALSE(has_max) << "Bound should no longer contain max, got: " << new_cond->__str__();

    // Check that the new bound is just N
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*new_cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(new_cond);
        EXPECT_TRUE(symbolic::eq(lt->get_arg2(), symbolic::symbol("N")))
            << "Expected bound to be N, got: " << lt->get_arg2()->__str__();
    }
}
