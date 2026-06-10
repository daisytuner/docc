/**
 * @file sets.h
 * @brief Integer set operations for symbolic expressions
 *
 * This file provides operations for interpreting symbolic expressions as integer sets
 * and computing their relations. It integrates with the ISL (Integer Set Library) to
 * perform set-theoretic operations on symbolic expressions.
 *
 * ## Integer Set Operations
 *
 * Symbolic expressions can be interpreted as sets of integer points.
 *
 * The primary operations are:
 * - **Subset checking**: Determining if one set of accesses is contained within another
 * - **Disjointness checking**: Verifying that two sets of accesses don't overlap
 *
 * These operations use assumptions about symbol bounds to compute precise set relations,
 * which is critical for dependence analysis and optimization.
 *
 * ## Example Usage
 *
 * @code
 * // Check if memory accesses A[i] and A[j] are disjoint when 0 <= i < 10 and 10 <= j < 20
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 *
 * MultiExpression expr1 = {i};  // Access pattern A[i]
 * MultiExpression expr2 = {j};  // Access pattern A[j]
 *
 * Assumptions assums1, assums2;
 * assums1[i].add_lower_bound(symbolic::zero());
 * assums1[i].add_upper_bound(symbolic::integer(10));
 * assums2[j].add_lower_bound(symbolic::integer(11));
 * assums2[j].add_upper_bound(symbolic::integer(20));
 *
 * bool disjoint = is_disjoint(expr1, expr2, assums1, assums2);  // true
 * @endcode
 *
 * @see assumptions.h for information about symbol assumptions
 * @see utils.h for ISL integration utilities
 */

#pragma once

#include <string>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is a subset of expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is a subset of expr2, false otherwise.
 */
bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
);

/**
 * @brief Same as `is_subset`, but reuses caller-supplied `AssumptionsBounds`.
 *
 * Constructing one `AssumptionsBounds` per scope and threading it through many
 * subset queries amortizes the internal `delinearize` bound-analysis cache
 * across all of them.
 */
bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, AssumptionsBounds& bounds1, AssumptionsBounds& bounds2
);

/**
 * @brief Interprets the expressions as integer sets and checks if expr1 is disjoint from expr2.
 *
 * @param expr1 The first expression to check.
 * @param expr2 The second expression to check.
 * @param assums1 The assumptions for the first expression.
 * @param assums2 The assumptions for the second expression.
 * @return true if expr1 is disjoint from expr2, false otherwise.
 */
bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
);

/**
 * @brief Same as `is_disjoint`, but reuses caller-supplied `AssumptionsBounds`.
 */
bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, AssumptionsBounds& bounds1, AssumptionsBounds& bounds2
);

} // namespace symbolic
} // namespace sdfg
