/**
 * @file polynomials.h
 * @brief Polynomial representation and manipulation
 *
 * This file provides utilities for converting symbolic expressions to polynomial
 * form and extracting coefficient information.
 *
 * ## Polynomial Representation
 *
 * Many symbolic expressions in SDFGs are polynomial in nature. This module
 * provides functions to convert expressions to polynomial form
 * and extract structural information.
 *
 * Affine expressions (degree 1 polynomials) are particularly important as they
 * represent linear transformations commonly found in loop index calculations
 * and memory addressing.
 *
 * ## Example Usage
 *
 * @code
 * // Convert an expression to polynomial form and extract coefficients
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * auto expr = symbolic::add(symbolic::mul(symbolic::integer(2), i),
 *                          symbolic::add(symbolic::mul(symbolic::integer(3), j),
 *                                       symbolic::integer(5)));  // 2*i + 3*j + 5
 *
 * SymbolVec symbols = {i, j};
 * auto poly = polynomial(expr, symbols);
 * auto coeffs = affine_coefficients(poly, symbols);
 * // coeffs[i] = 2, coeffs[j] = 3, constant = 5
 * @endcode
 *
 * @see symbolic.h for building symbolic expressions
 */

#pragma once

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/** @brief Type representing a SymEngine polynomial expression */
typedef SymEngine::RCP<const SymEngine::MExprPoly> Polynomial;

/** @brief Map from symbols to their coefficients in an affine expression */
typedef std::unordered_map<Symbol, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> AffineCoeffs;

/**
 * @brief Converts an Expression to a Polynomial
 *
 * @param expr The Expression to convert
 * @param symbols A vector of symbols that will be used in the polynomial
 *
 * @return A Polynomial representation of the Expression
 *
 * Converts a symbolic expression into SymEngine's polynomial representation,
 * which allows for efficient polynomial operations and coefficient extraction.
 * The symbols vector specifies which symbols should be treated as polynomial
 * variables.
 */
Polynomial polynomial(const Expression expr, SymbolVec& symbols);

/**
 * @brief Converts a Polynomial of degree 1 to coefficient map
 * @param poly The Polynomial to convert (must be degree 1)
 * @param symbols A vector of symbols that will be used in the coefficients map
 *
 * @return A AffineCoeffs map where keys are symbols and values are their coefficients
 *
 * Extracts the coefficients from a degree-1 (affine) polynomial. For an expression
 * like 2*i + 3*j + 5, this returns a map with entries {i: 2, j: 3} plus the constant term.
 *
 * @code
 * // Extract coefficients from 2*i + 3*j + 5
 * SymbolVec symbols = {i, j};
 * auto poly = polynomial(expr, symbols);
 * auto coeffs = affine_coefficients(poly, symbols);
 * // Access individual coefficients: coeffs[i] gives 2
 * @endcode
 */
AffineCoeffs affine_coefficients(Polynomial poly, SymbolVec& symbols);

/**
 * @brief Computes the inverse function for an affine expression
 * @param coeffs Coefficient map from affine_coefficients()
 * @param symbol The symbol to solve for
 * @return Expression representing the inverse function
 *
 * Given an affine expression y = a*x + b (represented as coefficients),
 * computes the inverse x = (y - b) / a.
 *
 * @code
 * // If we have y = 2*i + 5, compute i in terms of y
 * auto inv = affine_inverse(coeffs, i);  // Returns (y - 5) / 2
 * @endcode
 */
Expression affine_inverse(AffineCoeffs coeffs, Symbol symbol);

/**
 * @brief Result of affine decomposition of an expression w.r.t. a symbol
 *
 * For an expression `expr = coeff * symbol + offset`, this struct holds
 * the coefficient and offset values.
 */
struct AffineDecomposition {
    bool success; ///< True if the expression is affine in the symbol
    Expression coeff; ///< Coefficient of the symbol (must be non-zero integer for success)
    Expression offset; ///< Constant offset (terms not involving the symbol)

    static AffineDecomposition failure() { return {false, SymEngine::null, SymEngine::null}; }
};

/**
 * @brief Decomposes an expression into affine form w.r.t. a single symbol
 * @param expr The expression to decompose
 * @param symbol The symbol to decompose with respect to
 * @return AffineDecomposition containing coefficient and offset, or failure
 *
 * Given an expression, checks if it is affine (degree <= 1) in the given symbol.
 * If so, returns the coefficient and offset such that expr = coeff * symbol + offset.
 *
 * @code
 * auto result = affine_decomposition(parse("-2 + _s0"), symbol("_s0"));
 * // result.success = true, result.coeff = 1, result.offset = -2
 *
 * auto result2 = affine_decomposition(parse("_s0 * _s0"), symbol("_s0"));
 * // result2.success = false (quadratic, not affine)
 * @endcode
 */
AffineDecomposition affine_decomposition(const Expression& expr, const Symbol& symbol);

/**
 * @brief Solves for a symbol's bound given an inequality constraint
 * @param expr Expression containing the symbol
 * @param symbol The symbol to solve for
 * @param bound_value The value that expr must be >= (for lower) or <= (for upper)
 * @param is_lower_bound True if solving expr >= bound_value, false for expr <= bound_value
 * @return The bound on the symbol, or null if cannot be solved
 *
 * Given an affine expression `expr = coeff * symbol + offset` and a constraint:
 * - For lower bound (expr >= bound_value): symbol >= (bound_value - offset) / coeff (if coeff > 0)
 * - For upper bound (expr <= bound_value): symbol <= (bound_value - offset) / coeff (if coeff > 0)
 *
 * Only works when coeff is a positive integer.
 *
 * @code
 * // For loop bound: -2 + _s0 >= 1 (loop must execute at least once)
 * auto bound = solve_affine_bound(parse("-2 + _s0"), symbol("_s0"), integer(1), true);
 * // Returns 3 (_s0 >= 3)
 * @endcode
 */
Expression
solve_affine_bound(const Expression& expr, const Symbol& symbol, const Expression& bound_value, bool is_lower_bound);

/**
 * @brief Performs polynomial division
 * @param dividend The polynomial to divide
 * @param divisor The polynomial to divide by
 * @return Pair of (quotient, remainder) expressions
 *
 * Performs polynomial division and returns both the quotient and remainder.
 * For polynomials p(x) and q(x), computes p(x) = q(x)*quotient + remainder.
 *
 * @code
 * auto dividend = parse("x*x + 3*x + 2");  // x^2 + 3x + 2
 * auto divisor = parse("x + 1");            // x + 1
 * auto [quot, rem] = polynomial_div(dividend, divisor);
 * // quot = x + 2, rem = 0
 * @endcode
 */
std::pair<Expression, Expression> polynomial_div(const Expression& dividend, const Expression& divisor);


} // namespace symbolic
} // namespace sdfg
