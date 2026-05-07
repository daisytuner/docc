#include "sdfg/symbolic/extreme_values.h"

#include <gtest/gtest.h>
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

// ===== Tests for minimum / maximum =====

TEST(ExtremeValuesTest, Symbol_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(lb);
    assum.add_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum(a, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum(a, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesTest, Symbol_Integral_Tight) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.tight_lower_bound(lb);
    assum.tight_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum(a, {}, assums, true);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum(a, {}, assums, true);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesTest, Symbol_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(N);
    assum_a.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});

    auto min = symbolic::minimum(a, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, N));

    auto max = symbolic::maximum(a, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, M));
}

TEST(ExtremeValuesTest, Linear_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(lb);
    assum.add_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(13)));
}

TEST(ExtremeValuesTest, Linear_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(N);
    assum.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, {N, M}, assums, false);
    auto expr_lb = symbolic::
        add(symbolic::min(symbolic::mul(symbolic::integer(4), M), symbolic::mul(symbolic::integer(4), N)),
            symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(min, expr_lb));

    auto max = symbolic::maximum(expr, {N, M}, assums, false);
    auto expr_ub = symbolic::
        add(symbolic::max(symbolic::mul(symbolic::integer(4), M), symbolic::mul(symbolic::integer(4), N)),
            symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(max, expr_ub));
}

TEST(ExtremeValuesTest, Max_Integral) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto lb_a = symbolic::integer(1);
    auto ub_a = symbolic::integer(2);

    auto lb_b = symbolic::integer(3);
    auto ub_b = symbolic::integer(4);

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(lb_a);
    assum_a.add_upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.add_lower_bound(lb_b);
    assum_b.add_upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    // min(max(a,b)) = max(min(a), min(b)) = max(1, 3) = 3
    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(3)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesTest, Max_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(N);
    assum_a.add_upper_bound(M);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.add_lower_bound(N_);
    assum_b.add_upper_bound(M_);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    // min(max(a,b)) = max(min(a), min(b)) = max(N, N')
    auto min = symbolic::minimum(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::max(N, N_)));

    auto max = symbolic::maximum(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}

TEST(ExtremeValuesTest, Min_Integral) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto lb_a = symbolic::integer(1);
    auto ub_a = symbolic::integer(2);

    auto lb_b = symbolic::integer(3);
    auto ub_b = symbolic::integer(4);

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(lb_a);
    assum_a.add_upper_bound(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.add_lower_bound(lb_b);
    assum_b.add_upper_bound(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    // max(min(a,b)) = min(max(a), max(b)) = min(2, 4) = 2
    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(2)));
}

TEST(ExtremeValuesTest, Min_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(N);
    assum_a.add_upper_bound(M);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.add_lower_bound(N_);
    assum_b.add_upper_bound(M_);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    // max(min(a,b)) = min(max(a), max(b)) = min(M, M')
    auto max = symbolic::maximum(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::min(M, M_)));
}

TEST(ExtremeValuesTest, Recursive_Assumptions) {
    auto i = symbolic::symbol("i");
    auto i_init = symbolic::symbol("i_init");
    auto i_end_ex = symbolic::symbol("i_end_ex");
    auto j = symbolic::symbol("j");
    auto j_init = symbolic::symbol("j_init");

    symbolic::Assumption assum_i = symbolic::Assumption(i);
    assum_i.add_lower_bound(symbolic::symbol("i_init"));
    assum_i.add_upper_bound(symbolic::symbol("i_end_ex"));

    symbolic::Assumption assum_i_init = symbolic::Assumption(i_init);
    assum_i_init.add_lower_bound(symbolic::integer(0));
    assum_i_init.add_upper_bound(symbolic::integer(0));

    symbolic::Assumption assum_j = symbolic::Assumption(j);
    assum_j.add_lower_bound(symbolic::symbol("j_init"));
    assum_j.add_upper_bound(symbolic::symbol("j_end_ex"));

    auto assumptions = symbolic::Assumptions{{i, assum_i}, {i_init, assum_i_init}, {j, assum_j}};

    auto parameters = symbolic::SymbolSet{i_end_ex};

    auto i_min = symbolic::minimum(i, parameters, assumptions, false);
    EXPECT_TRUE(symbolic::eq(i_min, symbolic::integer(0)));
    auto i_max = symbolic::maximum(i, parameters, assumptions, false);
    EXPECT_TRUE(symbolic::eq(i_max, i_end_ex));

    auto j_min = symbolic::minimum(j, parameters, assumptions, false);
    EXPECT_TRUE(j_min.is_null());
}

// ===== Tests for idiv and imod operations =====

TEST(ExtremeValuesTest, IDiv_Integral) {
    // a in [4, 20], compute min/max of a / 4
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(4));
    assum.add_upper_bound(symbolic::integer(20));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(4));

    // min(a/4) = min(a) / max(4) = 4/4 = 1
    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    // max(a/4) = max(a) / min(4) = 20/4 = 5
    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(5)));
}

TEST(ExtremeValuesTest, IDiv_IntegralTruncation) {
    // a in [5, 17], compute min/max of a / 3 (integer division truncates)
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(5));
    assum.add_upper_bound(symbolic::integer(17));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(3));

    // min(a/3) = 5/3 = 1 (integer division)
    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    // max(a/3) = 17/3 = 5 (integer division)
    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(5)));
}

TEST(ExtremeValuesTest, IDiv_Symbolic) {
    // a in [N, M], compute min/max of a / 4
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(N);
    assum.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(4));

    // min(a/4) = N/4
    auto min = symbolic::minimum(expr, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::div(N, symbolic::integer(4))));

    // max(a/4) = M/4
    auto max = symbolic::maximum(expr, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::div(M, symbolic::integer(4))));
}

TEST(ExtremeValuesTest, IDiv_DivisorOne) {
    // a in [3, 7], a / 1 should simplify to a
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(3));
    assum.add_upper_bound(symbolic::integer(7));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(1));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(3)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(7)));
}

TEST(ExtremeValuesTest, IDiv_NonConstDenominator) {
    // a in [4, 20], b in [2, 5], a / b -> denominator not constant integer -> null
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(symbolic::integer(4));
    assum_a.add_upper_bound(symbolic::integer(20));

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.add_lower_bound(symbolic::integer(2));
    assum_b.add_upper_bound(symbolic::integer(5));

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    // Force a symbolic idiv by using function_symbol directly
    auto expr = SymEngine::function_symbol("idiv", {a, b});

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(min.is_null());

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(max.is_null());
}

TEST(ExtremeValuesTest, IMod_Integral) {
    // a in [5, 17], compute min/max of a % 3
    // mod(a, 3) = a - (a/3)*3
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(5));
    assum.add_upper_bound(symbolic::integer(17));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    auto max = symbolic::maximum(expr, {}, assums, false);

    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(2)));
}

TEST(ExtremeValuesTest, IMod_Symbolic) {
    // a in [N, M], compute min/max of a % 4
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(N);
    assum.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(4));

    auto min = symbolic::minimum(expr, {N, M}, assums, false);
    auto max = symbolic::maximum(expr, {N, M}, assums, false);


    auto expected_min = symbolic::zero();
    EXPECT_TRUE(symbolic::eq(min, expected_min));
    auto expected_max = symbolic::integer(3);
    EXPECT_TRUE(symbolic::eq(max, expected_max));
}

TEST(ExtremeValuesTest, IMod_Integral_reduced) {
    // a in [0, 1], compute min/max of a % 3
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(0));
    assum.add_upper_bound(symbolic::integer(1));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    auto max = symbolic::maximum(expr, {}, assums, false);

    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(1)));
}

TEST(ExtremeValuesTest, Conv2d_Access_Pattern) {
    // Expression from generated conv2d code accessing padded input buffer:
    // k10 + 3364*(ic0 + 64*(idiv(idiv(oc0_collapsed0, 32), 128)))
    //     + 58*(k00 + idiv(od00_collapsed0, 56))
    //     + 215296*imod(oc0_collapsed0, 32)
    //     + imod(od00_collapsed0, 56)
    //
    // Loop bounds:
    //   k10 in [0, 2], k00 in [0, 2], ic0 in [0, 63]
    //   oc0_collapsed0 in [0, 4095], od00_collapsed0 in [0, 3135]
    //
    // Buffer size: 27557888 bytes = 6889472 floats
    // Expected: min = 0, max = 6889471

    auto k10 = symbolic::symbol("k10");
    auto k00 = symbolic::symbol("k00");
    auto ic0 = symbolic::symbol("ic0");
    auto oc0_collapsed0 = symbolic::symbol("oc0_collapsed0");
    auto od00_collapsed0 = symbolic::symbol("od00_collapsed0");

    symbolic::Assumption assum_k10(k10);
    assum_k10.add_lower_bound(symbolic::integer(0));
    assum_k10.add_upper_bound(symbolic::integer(2));

    symbolic::Assumption assum_k00(k00);
    assum_k00.add_lower_bound(symbolic::integer(0));
    assum_k00.add_upper_bound(symbolic::integer(2));

    symbolic::Assumption assum_ic0(ic0);
    assum_ic0.add_lower_bound(symbolic::integer(0));
    assum_ic0.add_upper_bound(symbolic::integer(63));

    symbolic::Assumption assum_oc0(oc0_collapsed0);
    assum_oc0.add_lower_bound(symbolic::integer(0));
    assum_oc0.add_upper_bound(symbolic::integer(4095));

    symbolic::Assumption assum_od0(od00_collapsed0);
    assum_od0.add_lower_bound(symbolic::integer(0));
    assum_od0.add_upper_bound(symbolic::integer(3135));

    symbolic::Assumptions assums;
    assums.insert({k10, assum_k10});
    assums.insert({k00, assum_k00});
    assums.insert({ic0, assum_ic0});
    assums.insert({oc0_collapsed0, assum_oc0});
    assums.insert({od00_collapsed0, assum_od0});

    // Build: k10 + 3364*(ic0 + 64*(idiv(idiv(oc0_collapsed0, 32), 128)))
    //          + 58*(k00 + idiv(od00_collapsed0, 56))
    //          + 215296*imod(oc0_collapsed0, 32)
    //          + imod(od00_collapsed0, 56)
    auto expr = symbolic::
        add(symbolic::
                add(symbolic::
                        add(symbolic::
                                add(k10,
                                    symbolic::
                                        mul(symbolic::integer(3364),
                                            symbolic::
                                                add(ic0,
                                                    symbolic::
                                                        mul(symbolic::integer(64),
                                                            symbolic::
                                                                div(symbolic::div(oc0_collapsed0, symbolic::integer(32)),
                                                                    symbolic::integer(128)))))),
                            symbolic::
                                mul(symbolic::integer(58),
                                    symbolic::add(k00, symbolic::div(od00_collapsed0, symbolic::integer(56))))),
                    symbolic::mul(symbolic::integer(215296), symbolic::mod(oc0_collapsed0, symbolic::integer(32)))),
            symbolic::mod(od00_collapsed0, symbolic::integer(56)));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(6889471)));
}

// ===== Tests for C++ truncation-toward-zero modulo with negative values =====

TEST(ExtremeValuesTest, IMod_NegativeDividend) {
    // a in [-7, -1], a % 3 (C++ semantics: result has sign of dividend)
    // Values: -7%3=-1, -6%3=0, -5%3=-2, -4%3=-1, -3%3=0, -2%3=-2, -1%3=-1
    // Expected: min = -2, max = 0
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-7));
    assum.add_upper_bound(symbolic::integer(-1));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesTest, IMod_NegativeDividend_SmallRange) {
    // a in [-2, -1], a % 3 (range < modulus)
    // Values: -2%3=-2, -1%3=-1
    // Expected: min = -2, max = -1
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-2));
    assum.add_upper_bound(symbolic::integer(-1));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(-1)));
}

TEST(ExtremeValuesTest, IMod_CrossingZero) {
    // a in [-3, 5], a % 4 (range crosses zero)
    // Values: -3%4=-3, -2%4=-2, -1%4=-1, 0%4=0, 1%4=1, 2%4=2, 3%4=3, 4%4=0, 5%4=1
    // Expected: min = -3, max = 3
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-3));
    assum.add_upper_bound(symbolic::integer(5));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(4));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-3)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(3)));
}

TEST(ExtremeValuesTest, IMod_NegativeDividend_ExactMultiple) {
    // a in [-6, -3], a % 3 (all are exact multiples or near)
    // Values: -6%3=0, -5%3=-2, -4%3=-1, -3%3=0
    // Expected: min = -2, max = 0
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-6));
    assum.add_upper_bound(symbolic::integer(-3));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesTest, IDiv_NegativeDividend) {
    // a in [-7, -1], a / 3 (C++ truncation toward zero)
    // Values: -7/3=-2, -6/3=-2, -5/3=-1, -4/3=-1, -3/3=-1, -2/3=0, -1/3=0
    // Expected: min = -2, max = 0
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-7));
    assum.add_upper_bound(symbolic::integer(-1));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesTest, IDiv_CrossingZero) {
    // a in [-5, 7], a / 3 (range crosses zero)
    // Values: -5/3=-1, -4/3=-1, -3/3=-1, -2/3=0, -1/3=0, 0/3=0, 1/3=0, ..., 7/3=2
    // Expected: min = -1, max = 2
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(-5));
    assum.add_upper_bound(symbolic::integer(7));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-1)));

    auto max = symbolic::maximum(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(2)));
}

TEST(ExtremeValuesTest, ConstantParameterInCoefficient) {
    // Regression test: constant parameters must stay in polynomial coefficients,
    // not become generators. Otherwise correlated terms like _i1*TSTEPS - 2*_i1
    // are split into separate monomials, producing unsound bounds.
    //
    // Expression: _i1*(TSTEPS-2) + _i2
    // where _i1, _i2 are iteration variables with maps, TSTEPS is a constant parameter.
    // _i1 in [0, 4], _i2 in [0, 4], TSTEPS has lower_bound=6 (constant, no upper bound).
    //
    // Expected: min = 0 * (6-2) + 0 = 0
    //           max can't be fully determined (TSTEPS has no upper bound),
    //           but min MUST be 0 (not negative).

    auto _i1 = symbolic::symbol("_i1");
    auto _i2 = symbolic::symbol("_i2");
    auto TSTEPS = symbolic::symbol("TSTEPS");

    // _i1: iteration variable with map and bounds [0, 4]
    symbolic::Assumption assum_i1(_i1);
    assum_i1.add_lower_bound(symbolic::integer(0));
    assum_i1.add_upper_bound(symbolic::integer(4));
    assum_i1.map(symbolic::add(_i1, symbolic::one()));
    assum_i1.constant(true);

    // _i2: iteration variable with map and bounds [0, 4]
    symbolic::Assumption assum_i2(_i2);
    assum_i2.add_lower_bound(symbolic::integer(0));
    assum_i2.add_upper_bound(symbolic::integer(4));
    assum_i2.map(symbolic::add(_i2, symbolic::one()));
    assum_i2.constant(true);

    // TSTEPS: constant parameter, lower_bound=6, no upper_bound, no map
    symbolic::Assumption assum_T(TSTEPS);
    assum_T.add_lower_bound(symbolic::integer(6));
    assum_T.constant(true);

    symbolic::Assumptions assums;
    assums.insert({_i1, assum_i1});
    assums.insert({_i2, assum_i2});
    assums.insert({TSTEPS, assum_T});

    // _i1*(TSTEPS-2) + _i2
    auto expr = symbolic::add(symbolic::mul(_i1, symbolic::sub(TSTEPS, symbolic::integer(2))), _i2);

    // Minimum must be 0: all terms are non-negative when _i1=0, _i2=0
    auto min_val = symbolic::minimum(expr, {}, assums, false);
    EXPECT_FALSE(min_val.is_null());
    EXPECT_TRUE(symbolic::eq(min_val, symbolic::integer(0)));
}

// ===== Inequality proofs =====

// Helper: build an Assumption with a single (lb, ub) range.
static symbolic::Assumption
make_range(const symbolic::Symbol& s, const symbolic::Expression& lb, const symbolic::Expression& ub) {
    symbolic::Assumption a(s);
    if (!lb.is_null()) a.add_lower_bound(lb);
    if (!ub.is_null()) a.add_upper_bound(ub);
    return a;
}

// ----- is_nonneg / is_positive on constants -----

TEST(InequalityProofs, IsNonneg_Constant_Zero) {
    symbolic::Assumptions empty;
    EXPECT_TRUE(symbolic::is_nonneg(symbolic::zero(), {}, empty, false));
    EXPECT_FALSE(symbolic::is_positive(symbolic::zero(), {}, empty, false));
}

TEST(InequalityProofs, IsNonneg_Constant_Positive) {
    symbolic::Assumptions empty;
    EXPECT_TRUE(symbolic::is_nonneg(symbolic::integer(7), {}, empty, false));
    EXPECT_TRUE(symbolic::is_positive(symbolic::integer(7), {}, empty, false));
}

TEST(InequalityProofs, IsNonneg_Constant_Negative) {
    symbolic::Assumptions empty;
    EXPECT_FALSE(symbolic::is_nonneg(symbolic::integer(-3), {}, empty, false));
    EXPECT_FALSE(symbolic::is_positive(symbolic::integer(-3), {}, empty, false));
    EXPECT_TRUE(symbolic::is_negative(symbolic::integer(-3), {}, empty, false));
    EXPECT_TRUE(symbolic::is_nonpos(symbolic::integer(-3), {}, empty, false));
}

// ----- is_nonneg via interval -----

TEST(InequalityProofs, IsNonneg_Symbol_With_LB_Zero) {
    auto i = symbolic::symbol("i");
    symbolic::Assumptions assums;
    assums.insert({i, make_range(i, symbolic::zero(), symbolic::integer(10))});

    EXPECT_TRUE(symbolic::is_nonneg(i, {}, assums, false));
    EXPECT_FALSE(symbolic::is_positive(i, {}, assums, false));
}

TEST(InequalityProofs, IsNonneg_Symbol_With_LB_One) {
    auto i = symbolic::symbol("i");
    symbolic::Assumptions assums;
    assums.insert({i, make_range(i, symbolic::one(), symbolic::integer(10))});

    EXPECT_TRUE(symbolic::is_nonneg(i, {}, assums, false));
    EXPECT_TRUE(symbolic::is_positive(i, {}, assums, false));
}

TEST(InequalityProofs, IsNonneg_AffineSum_AllPositive) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    symbolic::Assumptions assums;
    assums.insert({i, make_range(i, symbolic::zero(), symbolic::integer(5))});
    assums.insert({j, make_range(j, symbolic::zero(), symbolic::integer(5))});

    auto expr = symbolic::add(i, j);
    EXPECT_TRUE(symbolic::is_nonneg(expr, {}, assums, false));
}

// ----- Soundness: must NOT prove the unprovable -----

TEST(InequalityProofs, IsNonneg_Unrelated_NotProvable) {
    // expr = M - i with i, M >= 0 but no ordering between them.
    // Must NOT be provable (was a soundness bug in old strategy 3/4).
    auto i = symbolic::symbol("i");
    auto M = symbolic::symbol("M");
    symbolic::Assumptions assums;
    assums.insert({i, make_range(i, symbolic::zero(), SymEngine::null)});
    assums.insert({M, make_range(M, symbolic::zero(), SymEngine::null)});

    auto expr = symbolic::sub(M, i);
    EXPECT_FALSE(symbolic::is_nonneg(expr, {M}, assums, false));
}

TEST(InequalityProofs, IsNonneg_Related_BoundFromLT) {
    // expr = N - i with i in [0, N-1] (i.e. i < N). Should prove.
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({N, make_range(N, symbolic::one(), SymEngine::null)});
    auto N_minus_one = symbolic::sub(N, symbolic::one());
    assums.insert({i, make_range(i, symbolic::zero(), N_minus_one)});

    auto expr = symbolic::sub(N, i);
    EXPECT_TRUE(symbolic::is_nonneg(expr, {N}, assums, false));
    EXPECT_TRUE(symbolic::is_positive(expr, {N}, assums, false));
}

TEST(InequalityProofs, IsNonneg_Related_BoundFromLE) {
    // expr = N - i with i in [0, N]. Should prove >= 0 but NOT > 0.
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({N, make_range(N, symbolic::one(), SymEngine::null)});
    assums.insert({i, make_range(i, symbolic::zero(), N)});

    auto expr = symbolic::sub(N, i);
    EXPECT_TRUE(symbolic::is_nonneg(expr, {N}, assums, false));
    EXPECT_FALSE(symbolic::is_positive(expr, {N}, assums, false));
}

// ----- Max-aware lower bound (e.g. tight bounds shaped `c + max(0, X)`) -----

TEST(InequalityProofs, IsNonneg_OneePlusMax) {
    // 1 + max(0, X) >= 0 always.
    auto X = symbolic::symbol("X");
    symbolic::Assumptions empty;
    auto expr = symbolic::add(symbolic::one(), symbolic::max(symbolic::zero(), X));
    EXPECT_TRUE(symbolic::is_nonneg(expr, {X}, empty, false));
}

TEST(InequalityProofs, IsNonneg_MaxInTightBound) {
    // i has tight_lb = max(0, M-N), tight_ub = M.
    // Then `i >= 0` should be provable via Max descent on the tight LB.
    auto i = symbolic::symbol("i");
    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    symbolic::Assumption ai(i);
    ai.tight_lower_bound(symbolic::max(symbolic::zero(), symbolic::sub(M, N)));
    ai.tight_upper_bound(M);
    symbolic::Assumptions assums;
    assums.insert({i, ai});

    EXPECT_TRUE(symbolic::is_nonneg(i, {M, N}, assums, true));
}

// ----- Min-aware is_gt (the `stride > min(...)` pattern) -----

TEST(InequalityProofs, IsGt_StrideBeatsMin_AnyArg) {
    // stride = N; expr = i + min(N - i - 1, X)
    // For any i with i >= 0, N > i + (N - i - 1) = N - 1.
    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto X = symbolic::symbol("X");

    symbolic::Assumptions assums;
    assums.insert({N, make_range(N, symbolic::one(), SymEngine::null)});
    assums.insert({i, make_range(i, symbolic::zero(), symbolic::sub(N, symbolic::one()))});
    assums.insert({X, make_range(X, symbolic::zero(), SymEngine::null)});

    auto inner = symbolic::min(symbolic::sub(symbolic::sub(N, i), symbolic::one()), X);
    auto expr = symbolic::add(i, inner);
    EXPECT_TRUE(symbolic::is_gt(N, expr, {N, X}, assums, false));
}

// ----- Binary forms reduce to nonneg/positive -----

TEST(InequalityProofs, IsGe_IsLe_Symmetry) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    symbolic::Assumptions assums;
    assums.insert({a, make_range(a, symbolic::integer(5), symbolic::integer(10))});
    assums.insert({b, make_range(b, symbolic::integer(0), symbolic::integer(4))});

    EXPECT_TRUE(symbolic::is_ge(a, b, {}, assums, false));
    EXPECT_TRUE(symbolic::is_gt(a, b, {}, assums, false));
    EXPECT_TRUE(symbolic::is_le(b, a, {}, assums, false));
    EXPECT_TRUE(symbolic::is_lt(b, a, {}, assums, false));
    EXPECT_FALSE(symbolic::is_ge(b, a, {}, assums, false));
}

TEST(InequalityProofs, IsEq_Trivial) {
    auto a = symbolic::symbol("a");
    symbolic::Assumptions empty;
    EXPECT_TRUE(symbolic::is_eq(a, a, {}, empty, false));
    EXPECT_TRUE(symbolic::is_eq(symbolic::integer(7), symbolic::integer(7), {}, empty, false));
    EXPECT_FALSE(symbolic::is_eq(symbolic::integer(7), symbolic::integer(8), {}, empty, false));
}

// ----- LU-derived subscript shapes (regression coverage) -----
// These mirror the assumption shapes seen in the blocked-LU MLA test:
// outer i in [0, N), tile i_tile1 in [i_tile, i_tile + B - 1] but
// also clamped via min(N - 1, i_tile + B - 1).

TEST(InequalityProofs, LU_PanelIndex_NonNeg_KnownLimitation) {
    // `i_tile1 - i` with i in [i_tile, i_tile1]. The proof requires noticing
    // that `i_tile1` on both sides cancels — BoundAnalysis treats them as
    // independent occurrences. The delinearization-level upper-bound
    // substitution loop covers this case; the bare API does not.
    auto i = symbolic::symbol("i");
    auto i_tile = symbolic::symbol("i_tile");
    auto i_tile1 = symbolic::symbol("i_tile1");
    auto B = symbolic::integer(64);

    symbolic::Assumptions assums;
    assums.insert({i_tile, make_range(i_tile, symbolic::zero(), SymEngine::null)});
    assums.insert({
        i_tile1,
        make_range(i_tile1, i_tile, symbolic::sub(symbolic::add(i_tile, B), symbolic::one())),
    });
    assums.insert({i, make_range(i, i_tile, i_tile1)});

    auto expr = symbolic::sub(i_tile1, i);
    // Documents the limitation; do not change to EXPECT_TRUE without
    // first teaching `prove_ge_zero` to perform per-symbol substitution.
    EXPECT_FALSE(symbolic::is_nonneg(expr, {i_tile}, assums, false));
}

TEST(InequalityProofs, LU_PanelIndex_NonNeg_Independent) {
    // `i_tile + B - 1 - i` with i in [i_tile, i_tile + B - 1]. No coupling
    // through a shared symbol; the API can prove this.
    auto i = symbolic::symbol("i");
    auto i_tile = symbolic::symbol("i_tile");
    auto B = symbolic::integer(64);

    auto i_ub = symbolic::sub(symbolic::add(i_tile, B), symbolic::one());
    symbolic::Assumptions assums;
    assums.insert({i_tile, make_range(i_tile, symbolic::zero(), SymEngine::null)});
    assums.insert({i, make_range(i, i_tile, i_ub)});

    auto expr = symbolic::sub(i_ub, i);
    EXPECT_TRUE(symbolic::is_nonneg(expr, {i_tile}, assums, false));
}

TEST(InequalityProofs, LU_TileBound_StrideDominates) {
    // After upper-bound substitution (done by delinearization), the
    // `remaining` expression presented to `is_gt` may contain a Min:
    //   stride = N, r = i + min(N - i - 1, j_tile + B - 1)
    // The Min-descent path proves N > r by proving N > i + (N - i - 1) = N - 1.
    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j_tile = symbolic::symbol("j_tile");
    auto B = symbolic::integer(64);

    symbolic::Assumptions assums;
    assums.insert({N, make_range(N, symbolic::integer(2), SymEngine::null)});
    assums.insert({i, make_range(i, symbolic::zero(), symbolic::sub(N, symbolic::one()))});
    assums.insert({j_tile, make_range(j_tile, symbolic::zero(), SymEngine::null)});

    auto inner = symbolic::
        min(symbolic::sub(symbolic::sub(N, i), symbolic::one()),
            symbolic::sub(symbolic::add(j_tile, B), symbolic::one()));
    auto remaining = symbolic::add(i, inner);
    EXPECT_TRUE(symbolic::is_gt(N, remaining, {N}, assums, false));
}

// Diagnostic: documents the unsimplified shape that `minimum()` returns for
// `N - i` with `i in [0, N - 1]`. Helpful when debugging Min-descent paths.
TEST(InequalityProofs, Diagnostic_Minimum_NMinusI) {
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({N, make_range(N, symbolic::one(), SymEngine::null)});
    assums.insert({i, make_range(i, symbolic::zero(), symbolic::sub(N, symbolic::one()))});

    auto expr = symbolic::sub(N, i);
    auto lb = symbolic::minimum(expr, {N}, assums, false);
    EXPECT_FALSE(lb.is_null());
}
