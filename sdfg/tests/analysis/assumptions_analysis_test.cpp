#include "sdfg/analysis/assumptions_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(AssumptionsAnalysisTest, Init_bool) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("N", desc, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::one()));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 1);
    EXPECT_TRUE(analysis.is_parameter("N"));
}

TEST(AssumptionsAnalysisTest, Init_i8) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt8);
    types::Scalar desc_signed(types::PrimitiveType::Int8);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(255)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-128)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(127)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i16) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt16);
    types::Scalar desc_signed(types::PrimitiveType::Int16);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(65535)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-32768)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(32767)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i32) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt32);
    types::Scalar desc_signed(types::PrimitiveType::Int32);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(4294967295))
    );
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-2147483648))
    );
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(2147483647))
    );
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i64) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), SymEngine::Inf));
    EXPECT_TRUE(symbolic::
                    eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(),
                       symbolic::integer(std::numeric_limits<int64_t>::min())));
    EXPECT_TRUE(symbolic::
                    eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(),
                       symbolic::integer(std::numeric_limits<int64_t>::max())));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, For_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 2);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*i_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));

    auto& n_assumptions = assumptions.at(symbolic::symbol("N"));
    EXPECT_EQ(n_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(n_assumptions.upper_bounds().size(), 0);
    EXPECT_TRUE(n_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(n_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*n_assumptions.lower_bounds().begin(), symbolic::one()));
}

TEST(AssumptionsAnalysisTest, For_1D_And) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Le(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 3);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 2);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::
                    eq(i_assumptions.tight_upper_bound(), symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
    bool found_m = false;
    bool found_n = false;
    for (const auto& ub : i_assumptions.upper_bounds()) {
        if (symbolic::eq(ub, symbolic::symbol("N"))) {
            found_n = true;
        } else if (symbolic::eq(ub, symbolic::symbol("M"))) {
            found_m = true;
        }
    }
    EXPECT_TRUE(found_n);
    EXPECT_TRUE(found_m);
}

TEST(AssumptionsAnalysisTest, For_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);
    builder.add_container("j", desc_signed);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::one());
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    auto bound_2 = symbolic::symbol("N");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::add(indvar, symbolic::one());
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::one());

    auto& loop2 = builder.add_for(loop.root(), indvar_2, condition_2, init_2, update_2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop2.root());

    // Check
    EXPECT_EQ(assumptions.size(), 3);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(
        symbolic::eq(*i_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2)))
    );
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2)))
    );

    auto& j_assumptions = assumptions.at(symbolic::symbol("j"));
    EXPECT_EQ(j_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(j_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!j_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!j_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*j_assumptions.lower_bounds().begin(), symbolic::add(symbolic::symbol("i"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(j_assumptions.tight_lower_bound(), symbolic::add(symbolic::symbol("i"), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(*j_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(j_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));
}

// AssumptionsAnalysis propagates IfElse branch conditions into the body
// scope. Inside
//   for i in [0, N):
//     if (i >= 3 && i < N - 3): /* body */
// the body scope's assumption for `i` should add `i >= 3` and `i <= N - 4`
// (integer-domain conversion of `i < N - 3`) to the loop-derived bounds.
//
// Implementation: extract_assumptions_from_condition normalizes the
// condition to CNF (handling And/Or/Not/De Morgan uniformly), then for
// each single-literal conjunct emits per-symbol bounds via
// solve_affine_bound. Multi-literal (disjunctive) conjuncts are skipped
// because they do not soundly constrain a single symbol.
TEST(AssumptionsAnalysisTest, IfElse_BranchConditionPropagated) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("N", n_type, true);
    builder.add_container("i", n_type);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto three = symbolic::integer(3);
    auto four = symbolic::integer(4);

    // for i in [0, N): if (i >= 3 && i < N - 3) { /* taken */ }
    auto& loop = builder.add_for(root, i, symbolic::Lt(i, N), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& ife = builder.add_if_else(loop.root());
    auto cond = symbolic::And(symbolic::Ge(i, three), symbolic::Lt(i, symbolic::sub(N, three)));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assumptions = assumptions.at(i);

    // Loop-derived bounds are present (0 and N-1). The branch-derived bounds
    // (3 and N-4) are added alongside them. tight_lower_bound/tight_upper_bound
    // picks one bound from each set without resolving the "tightest" across
    // symbolic candidates — we just assert both old and new are in the set.
    bool has_three_lb = false;
    bool has_zero_lb = false;
    for (const auto& lb : i_assumptions.lower_bounds()) {
        if (symbolic::eq(lb, three)) has_three_lb = true;
        if (symbolic::eq(lb, symbolic::zero())) has_zero_lb = true;
    }
    EXPECT_TRUE(has_zero_lb) << "Loop-derived lower bound `0` should still be present.";
    EXPECT_TRUE(has_three_lb) << "Branch condition `i >= 3` should add `3` as a lower bound of `i`.";

    bool has_N_minus_4_ub = false;
    bool has_N_minus_1_ub = false;
    auto N_minus_4 = symbolic::sub(N, four);
    auto N_minus_1 = symbolic::sub(N, symbolic::one());
    for (const auto& ub : i_assumptions.upper_bounds()) {
        if (symbolic::eq(ub, N_minus_4)) has_N_minus_4_ub = true;
        if (symbolic::eq(ub, N_minus_1)) has_N_minus_1_ub = true;
    }
    EXPECT_TRUE(has_N_minus_1_ub) << "Loop-derived upper bound `N - 1` should still be present.";
    EXPECT_TRUE(has_N_minus_4_ub)
        << "Branch condition `i < N - 3` (integer domain) should add `N - 4` as an upper bound of `i`.";
}

// CNF coverage: a disjunctive guard does NOT yield per-symbol bounds, since
// `(i >= 5 OR i < 0)` does not soundly constrain `i`.
TEST(AssumptionsAnalysisTest, IfElse_DisjunctionNotPropagated) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("N", n_type, true);
    builder.add_container("i", n_type);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");

    // for i in [0, N): if (i >= 5 || i < 0) { /* taken */ }
    auto& loop = builder.add_for(root, i, symbolic::Lt(i, N), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& ife = builder.add_if_else(loop.root());
    auto cond = symbolic::Or(symbolic::Ge(i, symbolic::integer(5)), symbolic::Lt(i, symbolic::zero()));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assumptions = assumptions.at(i);

    // Only the loop-derived bounds remain — the disjunctive guard
    // (5 OR negative) is correctly skipped because no single conjunct
    // narrows `i`.
    for (const auto& lb : i_assumptions.lower_bounds()) {
        EXPECT_FALSE(symbolic::eq(lb, symbolic::integer(5)))
            << "Disjunctive guard `i >= 5 || i < 0` unsoundly tightened the lower bound of `i`.";
    }
}

// CNF coverage: a negated conjunction `!(i >= 3 && i < N - 3)` expands by
// De Morgan to `(i < 3 OR i >= N - 3)` — a single disjunctive clause that
// must NOT be split into bounds.
TEST(AssumptionsAnalysisTest, IfElse_NegatedConjunctionNotPropagated) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("N", n_type, true);
    builder.add_container("i", n_type);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto three = symbolic::integer(3);

    auto& loop = builder.add_for(root, i, symbolic::Lt(i, N), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& ife = builder.add_if_else(loop.root());
    auto inner_cond = symbolic::And(symbolic::Ge(i, three), symbolic::Lt(i, symbolic::sub(N, three)));
    auto cond = symbolic::Not(inner_cond);
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assumptions = assumptions.at(i);

    // Negation distributes to a disjunction — no per-symbol bound is sound.
    for (const auto& lb : i_assumptions.lower_bounds()) {
        EXPECT_FALSE(symbolic::eq(lb, three)) << "Negated conjunction unsoundly tightened `i`'s lower bound.";
    }
    for (const auto& ub : i_assumptions.upper_bounds()) {
        EXPECT_FALSE(symbolic::eq(ub, symbolic::sub(N, symbolic::integer(4))))
            << "Negated conjunction unsoundly tightened `i`'s upper bound.";
    }
}

// ===========================================================================
// Coupled-affine constraints emitted by IfElse guards over multiple indvars
// ===========================================================================

// Helper: check whether `set` contains an expression value-equal to `target`.
static bool contains_expr(const symbolic::ExpressionSet& set, const symbolic::Expression& target) {
    for (const auto& e : set) {
        if (symbolic::eq(e, target)) return true;
    }
    return false;
}

// A coupled guard `i + j <= 15` inside two nested loops registers the
// constraint `i + j - 15` on BOTH indvars' `constraints()` sets, while
// leaving per-symbol upper_bounds unchanged (the coupling cannot be
// expressed there without re-introducing cycles in BoundAnalysis).
TEST(AssumptionsAnalysisTest, IfElse_Coupled_LessEqual_RegistersConstraintOnBothIndvars) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [0, 10): for j in [0, 20): if (i + j <= 15) {}
    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(20)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    auto cond = symbolic::Le(symbolic::add(i, j), symbolic::integer(15));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assum = assumptions.at(i);
    auto& j_assum = assumptions.at(j);

    // Canonical form: `i + j - 15 <= 0` -> stored as `i + j - 15`.
    auto expected = symbolic::expand(symbolic::sub(symbolic::add(i, j), symbolic::integer(15)));

    EXPECT_TRUE(contains_expr(i_assum.constraints(), expected))
        << "Constraint `i + j - 15` should be registered on i's constraints().";
    EXPECT_TRUE(contains_expr(j_assum.constraints(), expected))
        << "Constraint `i + j - 15` should be registered on j's constraints().";

    // The coupled guard must NOT pollute per-symbol bounds.
    for (const auto& ub : i_assum.upper_bounds()) {
        EXPECT_FALSE(symbolic::eq(ub, symbolic::sub(symbolic::integer(15), j)))
            << "Coupled guard leaked into i's per-symbol upper_bounds.";
    }
    for (const auto& ub : j_assum.upper_bounds()) {
        EXPECT_FALSE(symbolic::eq(ub, symbolic::sub(symbolic::integer(15), i)))
            << "Coupled guard leaked into j's per-symbol upper_bounds.";
    }
}

// A coupled guard `i + j >= 5` registers `5 - i - j` on both indvars.
TEST(AssumptionsAnalysisTest, IfElse_Coupled_GreaterEqual_RegistersConstraintOnBothIndvars) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(20)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    // `i + j >= 5`  ===  `5 <= i + j`  — built as `Le(5, i+j)`.
    auto cond = symbolic::Le(symbolic::integer(5), symbolic::add(i, j));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assum = assumptions.at(i);
    auto& j_assum = assumptions.at(j);

    // Canonical form: `Le(5, i+j)` is `5 - (i+j) <= 0` -> stored as `5 - i - j`.
    auto expected = symbolic::expand(symbolic::sub(symbolic::integer(5), symbolic::add(i, j)));

    EXPECT_TRUE(contains_expr(i_assum.constraints(), expected))
        << "Constraint `5 - i - j` should be registered on i's constraints().";
    EXPECT_TRUE(contains_expr(j_assum.constraints(), expected))
        << "Constraint `5 - i - j` should be registered on j's constraints().";
}

// Conjunctive coupled guard: `5 <= i + j AND i + j <= 15` registers BOTH
// constraints on BOTH indvars (the halo-circle case from MLA tests).
TEST(AssumptionsAnalysisTest, IfElse_Coupled_Conjunction_RegistersBothConstraints) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(20)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    auto sum = symbolic::add(i, j);
    auto cond = symbolic::And(symbolic::Le(symbolic::integer(5), sum), symbolic::Le(sum, symbolic::integer(15)));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assum = assumptions.at(i);
    auto& j_assum = assumptions.at(j);

    auto upper = symbolic::expand(symbolic::sub(symbolic::add(i, j), symbolic::integer(15)));
    auto lower = symbolic::expand(symbolic::sub(symbolic::integer(5), symbolic::add(i, j)));

    EXPECT_EQ(i_assum.constraints().size(), 2u);
    EXPECT_EQ(j_assum.constraints().size(), 2u);
    EXPECT_TRUE(contains_expr(i_assum.constraints(), upper));
    EXPECT_TRUE(contains_expr(i_assum.constraints(), lower));
    EXPECT_TRUE(contains_expr(j_assum.constraints(), upper));
    EXPECT_TRUE(contains_expr(j_assum.constraints(), lower));
}

// Equality guard `i + j == K` registers TWO constraints `delta` and `-delta`
// (equality is the conjunction of `<= K` and `>= K`).
TEST(AssumptionsAnalysisTest, IfElse_Coupled_Equality_RegistersTwoConstraints) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(10)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    auto cond = symbolic::Eq(symbolic::add(i, j), symbolic::integer(8));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assum = assumptions.at(i);
    auto& j_assum = assumptions.at(j);

    auto upper = symbolic::expand(symbolic::sub(symbolic::add(i, j), symbolic::integer(8)));
    auto lower = symbolic::expand(symbolic::sub(symbolic::integer(8), symbolic::add(i, j)));

    EXPECT_EQ(i_assum.constraints().size(), 2u);
    EXPECT_EQ(j_assum.constraints().size(), 2u);
    EXPECT_TRUE(contains_expr(i_assum.constraints(), upper));
    EXPECT_TRUE(contains_expr(i_assum.constraints(), lower));
    EXPECT_TRUE(contains_expr(j_assum.constraints(), upper));
    EXPECT_TRUE(contains_expr(j_assum.constraints(), lower));
}

// A strict inequality `i + j < 16` is canonicalised to `i + j <= 15` on the
// integer domain (delta <= -1 -> delta + 1 <= 0 -> stored as `i + j - 15`).
TEST(AssumptionsAnalysisTest, IfElse_Coupled_StrictLess_TightensByOne) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(20)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    // `i + j < 16`  ->  delta = (i+j) - 16, K_le = -1, constraint = delta + 1 = i + j - 15.
    auto cond = symbolic::Lt(symbolic::add(i, j), symbolic::integer(16));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    auto& i_assum = assumptions.at(i);
    auto expected = symbolic::expand(symbolic::sub(symbolic::add(i, j), symbolic::integer(15)));
    EXPECT_TRUE(contains_expr(i_assum.constraints(), expected))
        << "Strict `<` on integer domain should canonicalise to `i + j - 15`.";
}

// Constraints registered at a parent scope are PROPAGATED into nested scopes
// via `propagate()` — querying the inner Sequence still exposes the coupled
// constraint registered at the outer IfElse case.
TEST(AssumptionsAnalysisTest, IfElse_Coupled_ConstraintsPropagateIntoChildScope) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(20)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    auto cond = symbolic::Le(symbolic::add(i, j), symbolic::integer(15));
    auto& taken = builder.add_case(ife, cond);
    // A nested sub-sequence inside the taken branch — the constraint should
    // still be visible here, having been carried down by `propagate()`.
    auto& nested_block = builder.add_block(taken);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& nested_assumptions = analysis.get(nested_block);

    auto& i_assum = nested_assumptions.at(i);
    auto& j_assum = nested_assumptions.at(j);

    auto expected = symbolic::expand(symbolic::sub(symbolic::add(i, j), symbolic::integer(15)));
    EXPECT_TRUE(contains_expr(i_assum.constraints(), expected))
        << "Coupled constraint registered at IfElse case must propagate into nested scope (i).";
    EXPECT_TRUE(contains_expr(j_assum.constraints(), expected))
        << "Coupled constraint registered at IfElse case must propagate into nested scope (j).";
}

// Single-indvar literals do NOT take the constraint path — they keep
// emitting per-symbol bounds. This guards against a regression where the
// multi-indvar code accidentally swallows the single-indvar case.
TEST(AssumptionsAnalysisTest, IfElse_SingleIndvar_DoesNotEmitConstraint) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);

    auto i = symbolic::symbol("i");

    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& ife = builder.add_if_else(loop.root());
    auto cond = symbolic::And(symbolic::Ge(i, symbolic::integer(3)), symbolic::Le(i, symbolic::integer(7)));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);
    auto& i_assum = assumptions.at(i);

    // Per-symbol bounds were emitted...
    bool has_three_lb = false;
    bool has_seven_ub = false;
    for (const auto& lb : i_assum.lower_bounds()) {
        if (symbolic::eq(lb, symbolic::integer(3))) has_three_lb = true;
    }
    for (const auto& ub : i_assum.upper_bounds()) {
        if (symbolic::eq(ub, symbolic::integer(7))) has_seven_ub = true;
    }
    EXPECT_TRUE(has_three_lb);
    EXPECT_TRUE(has_seven_ub);

    // ...and the constraints set is untouched.
    EXPECT_TRUE(i_assum.constraints().empty());
}

// A non-affine coupled guard (e.g. `i * j <= K`) must be REJECTED — we do
// not register a constraint we cannot project, because BoundAnalysis only
// supports affine inversion.
TEST(AssumptionsAnalysisTest, IfElse_Coupled_NonAffine_Rejected) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("i", n_type);
    builder.add_container("j", n_type);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(10)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& inner = builder.add_for(
        outer.root(), j, symbolic::Lt(j, symbolic::integer(10)), symbolic::zero(), symbolic::add(j, symbolic::one())
    );
    auto& ife = builder.add_if_else(inner.root());
    auto cond = symbolic::Le(symbolic::mul(i, j), symbolic::integer(20));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    analysis::AssumptionsAnalysis analysis(sdfg, /*with_branch_conditions=*/true);
    analysis.run(analysis_manager);
    auto& assumptions = analysis.get(taken);

    EXPECT_TRUE(assumptions.at(i).constraints().empty())
        << "Non-affine coupled guard must not be registered as a constraint on i.";
    EXPECT_TRUE(assumptions.at(j).constraints().empty())
        << "Non-affine coupled guard must not be registered as a constraint on j.";
}
