#include "sdfg/symbolic/polynomials.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(PolynomialsTest, Linear_1D) {
    auto x = symbolic::symbol("x");
    auto m = symbolic::integer(2);
    auto b = symbolic::integer(1);

    auto expr = symbolic::add(symbolic::mul(m, x), b);

    symbolic::SymbolVec vars = {x};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly);
    EXPECT_EQ(coeffs.size(), 2);
    EXPECT_TRUE(symbolic::eq(coeffs[x], m));
    EXPECT_TRUE(symbolic::eq(coeffs[symbolic::symbol("__daisy_constant__")], b));
}

TEST(PolynomialsTest, Linear_2D) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto m1 = symbolic::integer(2);
    auto b1 = symbolic::integer(1);
    auto m2 = symbolic::integer(3);
    auto b2 = symbolic::integer(4);

    auto expr = symbolic::add(symbolic::mul(m1, x), b1);
    expr = symbolic::add(expr, symbolic::mul(m2, y));
    expr = symbolic::add(expr, b2);

    symbolic::SymbolVec vars = {x, y};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly);
    EXPECT_EQ(coeffs.size(), 3);
    EXPECT_TRUE(symbolic::eq(coeffs[x], m1));
    EXPECT_TRUE(symbolic::eq(coeffs[y], m2));
    EXPECT_TRUE(symbolic::eq(coeffs[symbolic::symbol("__daisy_constant__")], symbolic::add(b1, b2)));
}

TEST(PolynomialsTest, Degree2) {
    auto x = symbolic::symbol("x");

    auto expr = symbolic::mul(x, x);

    symbolic::SymbolVec vars = {x};
    auto poly = symbolic::polynomial(expr, vars);

    auto coeffs = symbolic::affine_coefficients(poly);
    EXPECT_EQ(coeffs.size(), 0);
}

// Regression test: `polynomial()` inserts the caller's symbols into a sorted
// `ExpressionSet`, so the column ordering inside the resulting `MExprPoly` is
// determined by `RCPBasicKeyLess` (not by the caller's `SymbolVec` order).
// `affine_coefficients()` then iterates the polynomial's exponent vectors and
// maps `exponents[i]` back to `symbols[i]` as if the input order had been
// preserved — which yields wrong assignments whenever the two orders differ.
//
// We force the bug by constructing the same expression twice with mirrored
// `SymbolVec` orderings; both calls must produce the same coefficient map.
TEST(PolynomialsTest, Linear_2D_OrderIndependent) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");

    // expr = 3*x + 5*y + 7  (distinct, asymmetric coefficients so any swap
    // between the two columns is observable).
    auto expr = symbolic::
        add(symbolic::add(symbolic::mul(symbolic::integer(3), x), symbolic::mul(symbolic::integer(5), y)),
            symbolic::integer(7));

    symbolic::SymbolVec order_xy = {x, y};
    symbolic::SymbolVec order_yx = {y, x};

    auto poly_xy = symbolic::polynomial(expr, order_xy);
    auto poly_yx = symbolic::polynomial(expr, order_yx);

    auto coeffs_xy = symbolic::affine_coefficients(poly_xy);
    auto coeffs_yx = symbolic::affine_coefficients(poly_yx);

    auto k = symbolic::symbol("__daisy_constant__");

    EXPECT_TRUE(symbolic::eq(coeffs_xy[x], symbolic::integer(3))) << "xy: coeff(x)";
    EXPECT_TRUE(symbolic::eq(coeffs_xy[y], symbolic::integer(5))) << "xy: coeff(y)";
    EXPECT_TRUE(symbolic::eq(coeffs_xy[k], symbolic::integer(7))) << "xy: const";

    EXPECT_TRUE(symbolic::eq(coeffs_yx[x], symbolic::integer(3))) << "yx: coeff(x)";
    EXPECT_TRUE(symbolic::eq(coeffs_yx[y], symbolic::integer(5))) << "yx: coeff(y)";
    EXPECT_TRUE(symbolic::eq(coeffs_yx[k], symbolic::integer(7))) << "yx: const";
}


// Regression test: `polynomial_div` reconstructs monomials from the leading
// term's exponent vector using the caller's `SymbolVec`. The exponents are
// indexed by the polynomial's internal sorted variable order, so when the
// two orderings differ, divisions involving the "wrong" variable used to
// produce nonsensical quotients/remainders.
//
// Concrete witness: divide (a + b) by 1. The quotient must be (a + b) and
// the remainder zero, regardless of internal variable ordering. We probe by
// substituting concrete distinct values for `a` and `b` so any swap shows up
// numerically.
TEST(PolynomialsTest, PolynomialDiv_OrderIndependent) {
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    auto offset = symbolic::add(a, b);
    auto stride = symbolic::one();

    auto [q, r] = symbolic::polynomial_div(offset, stride);

    // Substitute a=2, b=11 into both q and offset; if a/b were swapped during
    // term reconstruction, q would still equal 13 (commutative add) — so use
    // a non-symmetric polynomial to expose the bug instead.
    auto offset2 = symbolic::add(symbolic::mul(symbolic::integer(3), a), symbolic::mul(symbolic::integer(7), b));
    auto [q2, r2] = symbolic::polynomial_div(offset2, stride);

    auto q2_at =
        symbolic::simplify(symbolic::subs(symbolic::subs(q2, a, symbolic::integer(2)), b, symbolic::integer(11)));
    // 3*2 + 7*11 = 83
    EXPECT_TRUE(symbolic::eq(q2_at, symbolic::integer(83))) << "q2 at (a=2,b=11): " << *q2_at;
    EXPECT_TRUE(symbolic::eq(symbolic::simplify(r2), symbolic::zero())) << "remainder: " << *r2;
}
