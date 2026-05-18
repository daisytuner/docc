/**
 * @file conjunctive_normal_form.h
 * @brief Conversion of conditions to conjunctive normal form (CNF)
 *
 * This file provides functionality for converting boolean conditions into
 * conjunctive normal form.
 *
 * ## Conjunctive Normal Form
 *
 * CNF is a standardized format for boolean expressions where the expression is
 * represented as a conjunction (AND) of clauses, where each clause is a disjunction
 * (OR) of literals. For example:
 *   (A OR B) AND (C OR D OR E) AND (F)
 *
 * ## Example Usage
 *
 * @code
 * auto x = symbolic::symbol("x");
 * auto y = symbolic::symbol("y");
 *
 * // Create a complex condition: (x > 0 AND y < 10) OR (x < 0 AND y > 10)
 * auto cond = symbolic::Or(
 *     symbolic::And(symbolic::Gt(x, zero()), symbolic::Lt(y, integer(10))),
 *     symbolic::And(symbolic::Lt(x, zero()), symbolic::Gt(y, integer(10)))
 * );
 *
 * // Convert to CNF
 * auto cnf = conjunctive_normal_form(cond);
 * // Result is a vector of clauses, each clause being a vector of conditions
 * @endcode
 *
 * @see symbolic.h for building boolean conditions
 */

#pragma once

#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @class CNFException
 * @brief Exception thrown when CNF conversion fails
 *
 * This exception is thrown when a condition cannot be converted to CNF,
 * typically due to unsupported operations or malformed expressions.
 */
class CNFException : public std::exception {
public:
    CNFException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

/**
 * @brief Conjunctive Normal Form representation
 *
 * A CNF is represented as a vector of clauses, where each clause is a vector
 * of conditions (literals). The overall expression is the AND of all clauses,
 * and each clause is the OR of its literals.
 *
 * Example: [[A, B], [C, D], [E]] represents (A OR B) AND (C OR D) AND E
 */
typedef std::vector<std::vector<Condition>> CNF;

/**
 * @brief Convert a condition to conjunctive normal form
 *
 * @param cond The condition to convert
 * @return The conjunctive normal form of the condition
 * @throws CNFException if the condition cannot be converted to CNF
 *
 * Converts a boolean condition into CNF by applying logical equivalences and
 * distribution laws. The result is a standardized form that can be used for
 * further analysis or optimization.
 *
 * @code
 * // Convert (A AND B) OR C to CNF
 * auto cnf = conjunctive_normal_form(condition);
 * // Result: (A OR C) AND (B OR C)
 * @endcode
 */
CNF conjunctive_normal_form(const Condition cond);

/**
 * @brief Check if a disjunctive clause is a tautology
 *
 * @param clause The clause to check, represented as a vector of conditions (literals)
 * @return true if the clause is a tautology, false otherwise
 */
bool is_tautology(const std::vector<symbolic::Condition>& clause);

} // namespace symbolic
} // namespace sdfg
