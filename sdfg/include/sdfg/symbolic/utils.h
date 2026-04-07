/**
 * @file utils.h
 * @brief ISL (Integer Set Library) integration utilities
 *
 * This file provides utilities for converting symbolic expressions to ISL format
 * and performing operations using the Integer Set Library.
 *
 * ## ISL Integration
 *
 * The Integer Set Library (ISL) is a tool for reasoning about integer sets
 * and relations. This module provides the bridge between sdfglib's symbolic expressions
 * and ISL's representation, enabling polyhedral analysis.
 *
 * @see sets.h for high-level set operations using ISL
 * @see assumptions.h for symbol assumptions used in constraints
 */

#pragma once

#include <isl/map.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Converts a multi-dimensional expression to an ISL map string
 * @param expr Multi-dimensional expression representing iteration or access space
 * @param assums Assumptions about symbol bounds
 * @return ISL-formatted string representing the expression as a map
 *
 * Converts symbolic expressions to ISL's string format for use with ISL library
 * functions. The resulting string includes constraints from assumptions.
 */
std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums);

/**
 * @brief Converts two expressions to ISL format for intersection analysis
 * @param expr1 First multi-dimensional expression
 * @param expr2 Second multi-dimensional expression
 * @param indvar Induction variable for the iteration space
 * @param assums1 Assumptions for first expression
 * @param assums2 Assumptions for second expression
 * @return Tuple of (map1_str, map2_str, combined_constraints_str) in ISL format
 *
 * Prepares two expressions and their assumptions for intersection checking using ISL.
 * Returns formatted strings that can be parsed by ISL to construct maps and perform
 * set operations.
 */
std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
);

/**
 * @brief Generates constraint expressions from assumptions
 * @param syms Set of symbols to generate constraints for
 * @param assums Assumptions containing bounds information
 * @param seen Set of symbols already processed (to avoid duplicates)
 * @return Set of constraint expressions derived from assumptions
 *
 * Extracts bound constraints from assumptions and converts them to constraint
 * expressions suitable for ISL. This includes lower bounds (sym >= lb) and
 * upper bounds (sym <= ub) for all symbols.
 */
ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

/**
 * @brief Converts a constraint expression to ISL string format
 * @param con Constraint expression (typically a comparison)
 * @return ISL-formatted string for the constraint
 *
 * Converts individual constraint expressions (like "i >= 0" or "i < N") to
 * ISL's string representation.
 */
std::string constraint_to_isl_str(const Expression con);

/**
 * @brief Canonicalizes dimension names in an ISL map
 * @param map ISL map to modify
 * @param in_prefix Prefix for input dimensions (e.g., "i")
 * @param out_prefix Prefix for output dimensions (e.g., "o")
 *
 * Renames the dimensions of an ISL map to use canonical names with specified prefixes.
 * This ensures consistent naming across different maps for easier composition and
 * comparison.
 */
void canonicalize_map_dims(isl_map* map, const std::string& in_prefix, const std::string& out_prefix);

} // namespace symbolic
} // namespace sdfg
