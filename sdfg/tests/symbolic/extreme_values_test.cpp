#include "sdfg/symbolic/extreme_values.h"

#include <gtest/gtest.h>
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(ExtremeValuesTest, Symbol_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum(a, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum(a, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesTest, Symbol_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound_deprecated(N);
    assum_a.upper_bound_deprecated(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});

    auto min = symbolic::minimum(a, {N, M}, assums);
    EXPECT_TRUE(symbolic::eq(min, N));

    auto max = symbolic::maximum(a, {N, M}, assums);
    EXPECT_TRUE(symbolic::eq(max, M));
}

TEST(ExtremeValuesTest, Linear_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(13)));
}

TEST(ExtremeValuesTest, Linear_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(N);
    assum.upper_bound_deprecated(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum(expr, {N, M}, assums);
    auto expr_lb = symbolic::
        add(symbolic::min(symbolic::mul(symbolic::integer(4), M), symbolic::mul(symbolic::integer(4), N)),
            symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(min, expr_lb));

    auto max = symbolic::maximum(expr, {N, M}, assums);
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
    assum_a.lower_bound_deprecated(lb_a);
    assum_a.upper_bound_deprecated(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound_deprecated(lb_b);
    assum_b.upper_bound_deprecated(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesTest, Max_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    auto lb_a = symbolic::symbol("N");
    auto ub_a = symbolic::symbol("M");

    auto lb_b = symbolic::symbol("N_");
    auto ub_b = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound_deprecated(lb_a);
    assum_a.upper_bound_deprecated(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound_deprecated(lb_b);
    assum_b.upper_bound_deprecated(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::max(a, b);

    auto min = symbolic::minimum(expr, {N, M, N_, M_}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum(expr, {N, M, N_, M_}, assums);
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
    assum_a.lower_bound_deprecated(lb_a);
    assum_a.upper_bound_deprecated(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound_deprecated(lb_b);
    assum_b.upper_bound_deprecated(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesTest, Min_Symbolic) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto N_ = symbolic::symbol("N_");
    auto M_ = symbolic::symbol("M_");

    auto lb_a = symbolic::symbol("N");
    auto ub_a = symbolic::symbol("M");

    auto lb_b = symbolic::symbol("N_");
    auto ub_b = symbolic::symbol("M_");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.lower_bound_deprecated(lb_a);
    assum_a.upper_bound_deprecated(ub_a);

    symbolic::Assumption assum_b = symbolic::Assumption(b);
    assum_b.lower_bound_deprecated(lb_b);
    assum_b.upper_bound_deprecated(ub_b);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});
    assums.insert({b, assum_b});

    auto expr = symbolic::min(a, b);

    auto min = symbolic::minimum(expr, {N, M, N_, M_}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum(expr, {N, M, N_, M_}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}

TEST(ExtremeValuesTest, Recursive_Assumptions) {
    auto i = symbolic::symbol("i");
    auto i_init = symbolic::symbol("i_init");
    auto i_end_ex = symbolic::symbol("i_end_ex");
    auto j = symbolic::symbol("j");
    auto j_init = symbolic::symbol("j_init");

    auto lb_i = symbolic::symbol("i_init");
    auto ub_i = symbolic::symbol("i_end_ex");

    auto lb_i_init = symbolic::integer(0);
    auto ub_i_init = symbolic::integer(0);

    auto lb_j = symbolic::symbol("j_init");
    auto ub_j = symbolic::symbol("j_end_ex");

    symbolic::Assumption assum_i = symbolic::Assumption(i);
    assum_i.lower_bound_deprecated(lb_i);
    assum_i.upper_bound_deprecated(ub_i);

    symbolic::Assumption assum_i_init = symbolic::Assumption(i_init);
    assum_i_init.lower_bound_deprecated(lb_i_init);
    assum_i_init.upper_bound_deprecated(ub_i_init);

    symbolic::Assumption assum_j = symbolic::Assumption(j);
    assum_j.lower_bound_deprecated(lb_j);
    assum_j.upper_bound_deprecated(ub_j);

    auto assumptions = symbolic::Assumptions{{i, assum_i}, {i_init, assum_i_init}, {j, assum_j}};

    auto parameters = symbolic::SymbolSet{i_end_ex};

    auto i_min = symbolic::minimum(i, parameters, assumptions);
    EXPECT_TRUE(symbolic::eq(i_min, symbolic::integer(0)));
    auto i_max = symbolic::maximum(i, parameters, assumptions);
    EXPECT_TRUE(symbolic::eq(i_max, i_end_ex));

    auto j_min = symbolic::minimum(j, parameters, assumptions);
    EXPECT_TRUE(j_min.is_null());
}

// ===== Tests for minimum_new / maximum_new =====

TEST(ExtremeValuesNewTest, Symbol_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(lb);
    assum.add_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum_new(a, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum_new(a, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesNewTest, Symbol_Integral_Tight) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.tight_lower_bound(lb);
    assum.tight_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto min = symbolic::minimum_new(a, {}, assums, true);
    EXPECT_TRUE(symbolic::eq(min, lb));

    auto max = symbolic::maximum_new(a, {}, assums, true);
    EXPECT_TRUE(symbolic::eq(max, ub));
}

TEST(ExtremeValuesNewTest, Symbol_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum_a = symbolic::Assumption(a);
    assum_a.add_lower_bound(N);
    assum_a.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum_a});

    auto min = symbolic::minimum_new(a, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, N));

    auto max = symbolic::maximum_new(a, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, M));
}

TEST(ExtremeValuesNewTest, Linear_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(lb);
    assum.add_upper_bound(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(9)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(13)));
}

TEST(ExtremeValuesNewTest, Linear_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(N);
    assum.add_upper_bound(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::add(symbolic::mul(symbolic::integer(4), a), symbolic::integer(5));

    auto min = symbolic::minimum_new(expr, {N, M}, assums, false);
    auto expr_lb = symbolic::
        add(symbolic::min(symbolic::mul(symbolic::integer(4), M), symbolic::mul(symbolic::integer(4), N)),
            symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(min, expr_lb));

    auto max = symbolic::maximum_new(expr, {N, M}, assums, false);
    auto expr_ub = symbolic::
        add(symbolic::max(symbolic::mul(symbolic::integer(4), M), symbolic::mul(symbolic::integer(4), N)),
            symbolic::integer(5));
    EXPECT_TRUE(symbolic::eq(max, expr_ub));
}

TEST(ExtremeValuesNewTest, Max_Integral) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesNewTest, Max_Symbolic) {
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

    auto min = symbolic::minimum_new(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum_new(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}

TEST(ExtremeValuesNewTest, Min_Integral) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(4)));
}

TEST(ExtremeValuesNewTest, Min_Symbolic) {
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

    auto min = symbolic::minimum_new(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::min(N, N_)));

    auto max = symbolic::maximum_new(expr, {N, M, N_, M_}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::max(M, M_)));
}

TEST(ExtremeValuesNewTest, Recursive_Assumptions) {
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

    auto i_min = symbolic::minimum_new(i, parameters, assumptions, false);
    EXPECT_TRUE(symbolic::eq(i_min, symbolic::integer(0)));
    auto i_max = symbolic::maximum_new(i, parameters, assumptions, false);
    EXPECT_TRUE(symbolic::eq(i_max, i_end_ex));

    auto j_min = symbolic::minimum_new(j, parameters, assumptions, false);
    EXPECT_TRUE(j_min.is_null());
}

// ===== Tests for idiv and imod operations =====

TEST(ExtremeValuesNewTest, IDiv_Integral) {
    // a in [4, 20], compute min/max of a / 4
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(4));
    assum.add_upper_bound(symbolic::integer(20));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(4));

    // min(a/4) = min(a) / max(4) = 4/4 = 1
    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    // max(a/4) = max(a) / min(4) = 20/4 = 5
    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(5)));
}

TEST(ExtremeValuesNewTest, IDiv_IntegralTruncation) {
    // a in [5, 17], compute min/max of a / 3 (integer division truncates)
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(5));
    assum.add_upper_bound(symbolic::integer(17));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(3));

    // min(a/3) = 5/3 = 1 (integer division)
    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    // max(a/3) = 17/3 = 5 (integer division)
    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(5)));
}

TEST(ExtremeValuesNewTest, IDiv_Symbolic) {
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
    auto min = symbolic::minimum_new(expr, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::div(N, symbolic::integer(4))));

    // max(a/4) = M/4
    auto max = symbolic::maximum_new(expr, {N, M}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::div(M, symbolic::integer(4))));
}

TEST(ExtremeValuesNewTest, IDiv_DivisorOne) {
    // a in [3, 7], a / 1 should simplify to a
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(3));
    assum.add_upper_bound(symbolic::integer(7));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::div(a, symbolic::integer(1));

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(3)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(7)));
}

TEST(ExtremeValuesNewTest, IDiv_NonConstDenominator) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(min.is_null());

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(max.is_null());
}

TEST(ExtremeValuesNewTest, IMod_Integral) {
    // a in [5, 17], compute min/max of a % 3
    // mod(a, 3) = a - (a/3)*3
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(5));
    assum.add_upper_bound(symbolic::integer(17));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    auto max = symbolic::maximum_new(expr, {}, assums, false);

    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(2)));
}

TEST(ExtremeValuesNewTest, IMod_Symbolic) {
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

    auto min = symbolic::minimum_new(expr, {N, M}, assums, false);
    auto max = symbolic::maximum_new(expr, {N, M}, assums, false);


    auto expected_min = symbolic::zero();
    EXPECT_TRUE(symbolic::eq(min, expected_min));
    auto expected_max = symbolic::integer(3);
    EXPECT_TRUE(symbolic::eq(max, expected_max));
}

TEST(ExtremeValuesNewTest, IMod_Integral_reduced) {
    // a in [0, 1], compute min/max of a % 3
    auto a = symbolic::symbol("a");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.add_lower_bound(symbolic::integer(0));
    assum.add_upper_bound(symbolic::integer(1));

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    auto expr = symbolic::mod(a, symbolic::integer(3));

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    auto max = symbolic::maximum_new(expr, {}, assums, false);

    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(1)));
}

TEST(ExtremeValuesNewTest, Conv2d_Access_Pattern) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(0)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(6889471)));
}

// ===== Tests for C++ truncation-toward-zero modulo with negative values =====

TEST(ExtremeValuesNewTest, IMod_NegativeDividend) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesNewTest, IMod_NegativeDividend_SmallRange) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(-1)));
}

TEST(ExtremeValuesNewTest, IMod_CrossingZero) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-3)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(3)));
}

TEST(ExtremeValuesNewTest, IMod_NegativeDividend_ExactMultiple) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesNewTest, IDiv_NegativeDividend) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-2)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(0)));
}

TEST(ExtremeValuesNewTest, IDiv_CrossingZero) {
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

    auto min = symbolic::minimum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(-1)));

    auto max = symbolic::maximum_new(expr, {}, assums, false);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(2)));
}
