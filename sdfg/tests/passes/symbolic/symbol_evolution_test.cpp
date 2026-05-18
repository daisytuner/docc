#include "sdfg/passes/symbolic/symbol_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// Positive case: sum = sum + 5, init = 0
// Expected closed-form: sum = 5*i
TEST(SymbolEvolutionTest, ConstantIncrement) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sum = sum + 5; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution is: sum = 5*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 5*i (since init is 0 and stride is 1)
        auto expected = symbolic::mul(symbolic::integer(5), i);
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

TEST(SymbolEvolutionTest, ConditionalConstantIncrement) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { if (i == 2) sum = sum + 5; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& if_block = builder.add_if_else(loop.root());
    auto& if_case = builder.add_case(if_block, symbolic::Eq(i, symbolic::integer(2)));
    builder.add_block(if_case, {{sum, symbolic::add(sum, symbolic::integer(5))}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Positive case: count = count - 3, init = 100
// Expected closed-form: count = 100 - 3*i
TEST(SymbolEvolutionTest, Pattern4_AffineUpdate_Subtraction) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("count", desc);
    auto i = symbolic::symbol("i");
    auto count = symbolic::symbol("count");

    auto& root = builder.subject().root();

    // Initialize: count = 100
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{count, symbolic::integer(100)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { count = count - 3; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{count, symbolic::sub(count, symbolic::integer(3))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution is: count = 100 - 3*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(count) != assignments.end());
    if (assignments.find(count) != assignments.end()) {
        auto evolved_expr = assignments.at(count);
        // Expected: 100 - 3*i
        auto expected = symbolic::sub(symbolic::integer(100), symbolic::mul(symbolic::integer(3), i));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Negative case: prod = prod * 2 (multiplicative, not affine)
TEST(SymbolEvolutionTest, Pattern4_AffineUpdate_Negative_Multiplication) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("prod", desc);
    auto i = symbolic::symbol("i");
    auto prod = symbolic::symbol("prod");

    auto& root = builder.subject().root();

    // Initialize: prod = 1
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{prod, symbolic::one()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { prod = prod * 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{prod, symbolic::mul(prod, symbolic::integer(2))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Positive case: c = 42 throughout
// Expected: c = 42
TEST(SymbolEvolutionTest, Pattern1_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("c", desc);
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");

    auto& root = builder.subject().root();

    // Initialize: c = 42
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{c, symbolic::integer(42)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { c = 42; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{c, symbolic::integer(42)}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: c = 42
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(c) != assignments.end());
    if (assignments.find(c) != assignments.end()) {
        auto evolved_expr = assignments.at(c);
        auto expected = symbolic::integer(42);
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Positive case: j tracks i exactly
// Expected: j = i
TEST(SymbolEvolutionTest, Pattern2_LoopAlias) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();

    // Initialize: j = 0 (same as i)
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{j, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { j = i + 1; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{j, symbolic::add(i, symbolic::one())}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: j = i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(j) != assignments.end());
    if (assignments.find(j) != assignments.end()) {
        auto evolved_expr = assignments.at(j);
        auto expected = i;
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// ============================================================================
// Negative Test Cases - Edge Cases
// ============================================================================

// Symbol is used after being updated
TEST(SymbolEvolutionTest, Negative_UsedAfterUpdate) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    builder.add_container("temp", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");
    auto temp = symbolic::symbol("temp");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sum = sum + 5; temp = sum; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block1, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block1.debug_info()
    );

    auto& loop_block2 = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block2, {{temp, sum}}, loop_block2.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Symbol is written multiple times
TEST(SymbolEvolutionTest, Negative_MultipleWrites) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("x", desc);
    auto i = symbolic::symbol("i");
    auto x = symbolic::symbol("x");

    auto& root = builder.subject().root();

    // Initialize: x = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{x, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { x = x + 1; x = x + 2; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block1 = builder.add_block(loop.root());
    builder
        .add_block_after(loop.root(), loop_block1, {{x, symbolic::add(x, symbolic::one())}}, loop_block1.debug_info());

    auto& loop_block2 = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block2, {{x, symbolic::add(x, symbolic::integer(2))}}, loop_block2.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// ============================================================================
// Extended Tests with Closed-Form Verification
// These tests verify that the pass produces the correct closed-form expression
// ============================================================================

// Test Pattern 3: Quadratic function sq = i*i
// Expected closed-form: sq = (i-1)*(i-1)
TEST(SymbolEvolutionTest, Pattern3_Quadratic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sq", desc);
    auto i = symbolic::symbol("i");
    auto sq = symbolic::symbol("sq");

    auto& root = builder.subject().root();

    // Initialize: sq = 0 (which is (-1)*(-1) = 1, but we need to match the pattern)
    // For Pattern 3 to work: init value should equal f(init - stride)
    // f(i) = i*i, so f(-1) = 1, but we're testing if init = 0 works
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sq, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sq = i*i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{sq, symbolic::mul(i, i)}}, loop_block.debug_info());

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Test Pattern 3: Linear function double_i = 2*i
// Expected closed-form: double_i = 2*(i-1)
// TODO: Pattern 3 linear function needs init condition adjustment
TEST(SymbolEvolutionTest, Pattern3_LinearFunction) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("double_i", desc);
    auto i = symbolic::symbol("i");
    auto double_i = symbolic::symbol("double_i");

    auto& root = builder.subject().root();

    // Initialize: double_i = -2 (which is 2*(0-1))
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{double_i, symbolic::integer(-2)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { double_i = 2*i; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{double_i, symbolic::mul(symbolic::integer(2), i)}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: double_i = 2*(i-1)
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(double_i) != assignments.end());
    if (assignments.find(double_i) != assignments.end()) {
        auto evolved_expr = assignments.at(double_i);
        // Expected: 2*(i-1)
        auto expected = symbolic::mul(symbolic::integer(2), symbolic::sub(i, symbolic::one()));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Test Pattern 4: Affine with offset
// sum = sum + 5, init = 10
// Expected: sum = 10 + 5*i
TEST(SymbolEvolutionTest, Pattern4_AffineWithOffset) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 10
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::integer(10)}}, init_block.debug_info());

    // Loop: for (i = 0; i < 10; i++) { sum = sum + 5; }
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify the closed-form solution: sum = 10 + 5*i
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 10 + 5*i
        auto expected = symbolic::add(symbolic::integer(10), symbolic::mul(symbolic::integer(5), i));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// Test verifying value after loop
// sum = sum + 3, for i=5 to i<10, stride=1
// Expected after loop: sum = init + 3*5 = init + 15
TEST(SymbolEvolutionTest, Pattern4_FinalValueVerification) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    // Initialize: sum = 0
    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    // Loop: for (i = 5; i < 10; i++) { sum = sum + 3; }
    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::integer(5), symbolic::add(i, symbolic::one())
    );

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(3))}}, loop_block.debug_info()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    // Verify: sum = 0 + 3*(i-5) = 3*i - 15
    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();

    EXPECT_TRUE(assignments.find(sum) != assignments.end());
    if (assignments.find(sum) != assignments.end()) {
        auto evolved_expr = assignments.at(sum);
        // Expected: 3*(i-5) = 3*i - 15
        auto expected = symbolic::mul(symbolic::integer(3), symbolic::sub(i, symbolic::integer(5)));
        EXPECT_TRUE(symbolic::eq(evolved_expr, expected));
    }
}

// ============================================================================
// Extended Tests for the unified affine recurrence solver
// ============================================================================

// Affine in indvar with non-unit slope: k = 3*i + 5, init = 2 (= 3*(0-1)+5)
// Expected closed form: k = 3*(i - 1) + 5
TEST(SymbolEvolutionTest, AffineInIndvar_NonUnit) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("k", desc);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{k, symbolic::integer(2)}}, init_block.debug_info());

    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(),
        loop_block,
        {{k, symbolic::add(symbolic::mul(symbolic::integer(3), i), symbolic::integer(5))}},
        loop_block.debug_info()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();
    ASSERT_TRUE(assignments.find(k) != assignments.end());
    auto evolved = assignments.at(k);
    auto expected =
        symbolic::add(symbolic::mul(symbolic::integer(3), symbolic::sub(i, symbolic::one())), symbolic::integer(5));
    auto diff = symbolic::simplify(symbolic::expand(symbolic::sub(evolved, expected)));
    EXPECT_TRUE(symbolic::eq(diff, symbolic::zero()));
}

// Negative stride accumulator: for (i = 10; i > 0; i--) sum += 1
// Expected closed form: sum = sum_init + 1*(i - 10)/(-1) = (10 - i)
TEST(SymbolEvolutionTest, NegativeStride_Accumulator) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    auto& loop =
        builder
            .add_for(root, i, symbolic::Gt(i, symbolic::zero()), symbolic::integer(10), symbolic::sub(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder
        .add_block_after(loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::one())}}, loop_block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();
    ASSERT_TRUE(assignments.find(sum) != assignments.end());
    auto evolved = assignments.at(sum);
    auto expected = symbolic::sub(symbolic::integer(10), i);
    // Compare by substitution at concrete points: structural simplification
    // of (10 - i) variants is fragile across SymEngine's normalization.
    for (int v : {1, 2, 5, 10}) {
        auto e = symbolic::simplify(symbolic::subs(evolved, i, symbolic::integer(v)));
        auto x = symbolic::simplify(symbolic::subs(expected, i, symbolic::integer(v)));
        EXPECT_TRUE(symbolic::eq(e, x)) << "at i=" << v;
    }
}

// Non-unit positive stride: for (i = 0; i < 20; i += 2) sum += 5
// Expected closed form: sum = 5 * (i / 2)
TEST(SymbolEvolutionTest, NonUnitStride_Accumulator) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(20)), symbolic::zero(), symbolic::add(i, symbolic::integer(2))
    );

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(
        loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::integer(5))}}, loop_block.debug_info()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    auto first_elem = loop.root().at(0);
    auto& assignments = first_elem.second.assignments();
    ASSERT_TRUE(assignments.find(sum) != assignments.end());
    auto evolved = assignments.at(sum);
    auto expected = symbolic::mul(symbolic::integer(5), symbolic::div(i, symbolic::integer(2)));
    auto diff = symbolic::simplify(symbolic::expand(symbolic::sub(evolved, expected)));
    EXPECT_TRUE(symbolic::eq(diff, symbolic::zero()));
}

// Constant body but mismatched init: c_init = 0, body c = 42.
// Closed form 42 fails the verification 42 == 0, so the rewrite must NOT apply.
TEST(SymbolEvolutionTest, Constant_DifferentInit_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("c", desc);
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{c, symbolic::zero()}}, init_block.debug_info());

    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& loop_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), loop_block, {{c, symbolic::integer(42)}}, loop_block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}

// Two independent accumulators in the same loop body. Each rewrite mutates
// the IR; the per-loop fixpoint must invalidate analyses and revisit the
// remaining candidate so that both end up rewritten in a single pass.
TEST(SymbolEvolutionTest, CoEvolving_FixpointRewrites) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    auto i = symbolic::symbol("i");
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{a, symbolic::zero()}, {b, symbolic::zero()}}, init_block.debug_info());

    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));

    auto& a_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), a_block, {{a, symbolic::add(a, symbolic::one())}}, a_block.debug_info());

    auto& b_block = builder.add_block(loop.root());
    builder.add_block_after(loop.root(), b_block, {{b, symbolic::add(b, symbolic::integer(2))}}, b_block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);

    bool a_rewritten = false;
    bool b_rewritten = false;
    for (size_t k = 0; k < loop.root().size(); ++k) {
        auto& tr = loop.root().at(k).second;
        if (tr.assignments().find(a) != tr.assignments().end()) a_rewritten = true;
        if (tr.assignments().find(b) != tr.assignments().end()) b_rewritten = true;
    }
    EXPECT_TRUE(a_rewritten);
    EXPECT_TRUE(b_rewritten);
}

// Nested loop where the inner update depends on the outer indvar:
//   for (i = 0; i < 5; i++) for (k = 0; k < 5; k++) { j = j + i; }
// `i` is loop-invariant w.r.t. the inner loop, so the accumulator solver
// produces closed(j) = j_init + i * (k - 0).
TEST(SymbolEvolutionTest, NestedLoop_OuterIndvarAsCoeff) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("k", desc);
    builder.add_container("j", desc);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{j, symbolic::zero()}}, init_block.debug_info());

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(5)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), k, symbolic::Lt(k, symbolic::integer(5)), symbolic::zero(), symbolic::add(k, symbolic::one())
    );

    auto& inner_block = builder.add_block(inner.root());
    builder.add_block_after(inner.root(), inner_block, {{j, symbolic::add(j, i)}}, inner_block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_TRUE(applied);
    auto& first_inner = inner.root().at(0).second;
    EXPECT_TRUE(first_inner.assignments().find(j) != first_inner.assignments().end());
}

// Symbolic (non-constant) stride: loop.stride() returns null, so the pass
// must conservatively skip.
TEST(SymbolEvolutionTest, SymbolicStride_Skipped) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("N", desc);
    builder.add_container("i", desc);
    builder.add_container("sum", desc);
    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto sum = symbolic::symbol("sum");

    auto& root = builder.subject().root();

    auto& init_block = builder.add_block(root);
    builder.add_block_after(root, init_block, {{sum, symbolic::zero()}}, init_block.debug_info());

    auto& loop =
        builder.add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, N));

    auto& loop_block = builder.add_block(loop.root());
    builder
        .add_block_after(loop.root(), loop_block, {{sum, symbolic::add(sum, symbolic::one())}}, loop_block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolEvolution pass;
    bool applied = pass.run(builder, analysis_manager);

    EXPECT_FALSE(applied);
}
