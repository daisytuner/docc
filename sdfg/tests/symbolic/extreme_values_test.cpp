#include "sdfg/symbolic/extreme_values.h"

#include <gtest/gtest.h>
#include <iostream>
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

TEST(ExtremeValuesTest, Pow_Integral) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(2);
    auto ub = symbolic::integer(3);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    // a^2 where a ∈ [2, 3]
    auto expr = symbolic::pow(a, symbolic::integer(2));

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(4))); // 2^2 = 4

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(9))); // 3^2 = 9
}

TEST(ExtremeValuesTest, Pow_Symbolic) {
    auto a = symbolic::symbol("a");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(N);
    assum.upper_bound_deprecated(M);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    // a^2 where a ∈ [N, M]
    auto expr = symbolic::pow(a, symbolic::integer(2));

    auto min = symbolic::minimum(expr, {N, M}, assums);
    auto expected_min = symbolic::pow(N, symbolic::integer(2));
    EXPECT_TRUE(symbolic::eq(min, expected_min)); // N^2

    auto max = symbolic::maximum(expr, {N, M}, assums);
    auto expected_max = symbolic::pow(M, symbolic::integer(2));
    EXPECT_TRUE(symbolic::eq(max, expected_max)); // M^2
}

TEST(ExtremeValuesTest, Pow_Exponent_Zero) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(2);
    auto ub = symbolic::integer(5);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    // a^0 = 1 for any a
    auto expr = symbolic::pow(a, symbolic::integer(0));

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1)));

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(1)));
}

TEST(ExtremeValuesTest, Pow_Exponent_One) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(3);
    auto ub = symbolic::integer(7);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    // a^1 = a
    auto expr = symbolic::pow(a, symbolic::integer(1));

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, lb)); // 3

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, ub)); // 7
}

TEST(ExtremeValuesTest, Pow_Higher_Exponent) {
    auto a = symbolic::symbol("a");

    auto lb = symbolic::integer(1);
    auto ub = symbolic::integer(2);

    symbolic::Assumption assum = symbolic::Assumption(a);
    assum.lower_bound_deprecated(lb);
    assum.upper_bound_deprecated(ub);

    symbolic::Assumptions assums;
    assums.insert({a, assum});

    // a^3 where a ∈ [1, 2]
    auto expr = symbolic::pow(a, symbolic::integer(3));

    auto min = symbolic::minimum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(min, symbolic::integer(1))); // 1^3 = 1

    auto max = symbolic::maximum(expr, {}, assums);
    EXPECT_TRUE(symbolic::eq(max, symbolic::integer(8))); // 2^3 = 8
}
