/**
 * @file maps.h
 * @brief Analysis of symbol evolution through maps
 *
 * This file provides functions for analyzing symbolic maps, as used in memory
 * accesses with respect to induction variables of loops.
 *
 * ## Map Analysis
 *
 * Key operations:
 * - **Monotonicity checking**: Determines if a map is strictly (positive) monotonic
 * - **Intersection checking**: Determines if the integer domains of two maps intersect
 *
 * @see assumptions.h for information about symbol assumptions and maps
 * @see symbolic.h for building symbolic expressions
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {
namespace maps {

/**
 * @brief Represents the set of dependence distance vectors between two access patterns.
 *
 * When two memory accesses touch overlapping cells at different iterations,
 * this struct captures the full set of iteration-distance vectors as an ISL set string.
 * The dimensions of the delta set correspond to the non-constant symbols
 * (loop induction variables) in the order listed in `dimensions`.
 */
struct DependenceDeltas {
    bool empty; ///< True if no cross-iteration dependence exists
    std::string deltas_str; ///< ISL set string representing the delta set (empty string if no dependence)
    std::vector<std::string> dimensions; ///< Dimension names in order, matching the ISL set dimensions
};

/**
 * @brief Checks if an expression is monotonic with respect to a symbol
 * @param expr The expression to check
 * @param sym The symbol to check monotonicity with respect to
 * @param assums Assumptions about symbols including the evolution map
 * @return true if expr is monotonic (always increasing or always decreasing) as sym evolves
 *
 * An expression is monotonic if it consistently increases as the symbol
 * evolves.
 */
bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums);

/**
 * @brief Checks if the integer domain of two maps intersect
 * @param expr1 First multi-dimensional expression (e.g., memory access pattern)
 * @param expr2 Second multi-dimensional expression
 * @param indvar The induction variable that evolves
 * @param assums1 Assumptions for the first expression including evolution maps
 * @param assums2 Assumptions for the second expression including evolution maps
 * @return true if the integer domains of expr1 and expr2 can overlap
 *
 * @code
 * // Check if A[i] and A[j+5] intersect when both evolve 0 to 10
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * MultiExpression expr1 = {i};
 * MultiExpression expr2 = {symbolic::add(j, symbolic::integer(5))};
 *
 * Assumptions assums1, assums2;
 * assums1[i].add_lower_bound(symbolic::zero());
 * assums1[i].add_upper_bound(symbolic::integer(10));
 * assums2[j].add_lower_bound(symbolic::zero());
 * assums2[j].add_upper_bound(symbolic::integer(10));
 *
 * bool overlap = intersects(expr1, expr2, i, assums1, assums2);  // true (e.g., i=7, j=2)
 * @endcode
 */
bool intersects(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
);

/**
 * @brief Computes the dependence distance delta set between two access patterns.
 *
 * Returns a DependenceDeltas struct containing the full ISL delta set
 * representing all possible iteration-distance vectors between aliasing
 * pairs of the two expressions. The dimensions correspond to all
 * non-constant symbols (induction variables).
 *
 * If the accesses are provably disjoint (via monotonicity or ISL analysis),
 * returns an empty delta set.
 */
DependenceDeltas dependence_deltas(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
);

/**
 * @brief Same as `dependence_deltas`, but reuses caller-supplied `AssumptionsBounds`.
 */
DependenceDeltas dependence_deltas(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    sdfg::symbolic::AssumptionsBounds& bounds1,
    sdfg::symbolic::AssumptionsBounds& bounds2
);

} // namespace maps
} // namespace symbolic
} // namespace sdfg
