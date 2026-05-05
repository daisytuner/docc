/**
 * @file assumptions.h
 * @brief Symbol assumptions for reasoning about bounds, constness, and evolution
 *
 * Assumptions are a critical component of the symbolic system that enable reasoning about
 * ranges of symbols, how they evolve, and whether they are constant. Examples:
 * - Ranges of memory accesses
 * - Ranges and evolution of loop iteration variables
 *
 * ## Assumption System
 *
 * Each symbol can have associated assumptions that describe its properties:
 * - **Bounds**: Lower and upper bounds that constrain the symbol's possible values
 * - **Tight bounds**: Exact minimum and maximum values the symbol can take
 * - **Constness**: Whether the symbol's value remains constant
 * - **Map**: How the symbol evolves (e.g., in loop iterations)
 *
 * ## Example Usage
 *
 * @code
 * // Create a symbol with type-based assumptions
 * auto i = symbolic::symbol("i");
 * auto int32_type = types::Scalar::create(types::PrimitiveType::Int32);
 * auto assumption = Assumption::create(i, int32_type);
 *
 * // Add loop bounds (e.g., 0 <= i < 10)
 * assumption.add_lower_bound(symbolic::zero());
 * assumption.add_upper_bound(symbolic::integer(10));
 *
 * // Mark as non-constant (evolves in loop)
 * assumption.constant(false);
 *
 * // Set evolution map (e.g., i' = i + 1)
 * assumption.map(symbolic::add(i, symbolic::one()));
 * @endcode
 *
 * @see extreme_values.h for computing bounds of expressions using assumptions
 * @see sets.h for integer set operations using assumptions
 */

#pragma once

#include <unordered_map>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace symbolic {

/**
 * @class Assumption
 * @brief Represents assumptions about a symbol's properties and behavior
 */
class Assumption {
private:
    Symbol symbol_;
    ExpressionSet lower_bounds_;
    ExpressionSet upper_bounds_;
    Expression tight_lower_bound_;
    Expression tight_upper_bound_;
    bool constant_;
    Expression map_;

public:
    /**
     * @brief Default constructor creating an empty assumption
     */
    Assumption();

    /**
     * @brief Constructs an assumption for a specific symbol
     * @param symbol The symbol this assumption is for
     */
    Assumption(const Symbol symbol);

    /**
     * @brief Copy constructor
     * @param a Assumption to copy
     */
    Assumption(const Assumption& a);

    /**
     * @brief Assignment operator
     * @param a Assumption to assign from
     * @return Reference to this assumption
     */
    Assumption& operator=(const Assumption& a);

    /**
     * @brief Gets the symbol this assumption is for
     * @return The symbol
     */
    const Symbol symbol() const;

    /**
     * @brief Gets all lower bound constraints
     * @return Set of all lower bound expressions
     *
     * A symbol may have multiple lower bound constraints (e.g., from different
     * analysis passes). The effective lower bound is the maximum of these.
     */
    const ExpressionSet& lower_bounds() const;

    /**
     * @brief Adds a lower bound constraint
     * @param lb The lower bound expression to add
     *
     * The symbol is constrained to be greater than or equal to this expression.
     * Multiple lower bounds can be added; the effective bound is their maximum.
     */
    void add_lower_bound(const Expression lb);

    /**
     * @brief Checks if a specific lower bound exists
     * @param lb The lower bound expression to check
     * @return true if this lower bound is present
     */
    bool contains_lower_bound(const Expression lb);

    /**
     * @brief Removes a lower bound constraint
     * @param lb The lower bound expression to remove
     * @return true if the bound was found and removed
     */
    bool remove_lower_bound(const Expression lb);

    /**
     * @brief Gets all upper bound constraints
     * @return Set of all upper bound expressions
     *
     * A symbol may have multiple upper bound constraints (e.g., from different
     * analysis passes). The effective upper bound is the minimum of these.
     */
    const ExpressionSet& upper_bounds() const;

    /**
     * @brief Adds an upper bound constraint
     * @param ub The upper bound expression to add
     *
     * The symbol is constrained to be less than or equal to this expression.
     * Multiple upper bounds can be added; the effective bound is their minimum.
     */
    void add_upper_bound(const Expression ub);

    /**
     * @brief Checks if a specific upper bound exists
     * @param ub The upper bound expression to check
     * @return true if this upper bound is present
     */
    bool contains_upper_bound(const Expression ub);

    /**
     * @brief Removes an upper bound constraint
     * @param ub The upper bound expression to remove
     * @return true if the bound was found and removed
     */
    bool remove_upper_bound(const Expression ub);

    /**
     * @brief Gets the tight lower bound
     * @return The exact minimum value the symbol can take, or null if unknown
     *
     * Unlike lower_bounds which are constraints, the tight lower bound represents
     * the exact minimum value the symbol will take during execution.
     */
    const Expression tight_lower_bound() const;

    /**
     * @brief Sets the tight lower bound
     * @param tight_lb The exact minimum value
     *
     * Sets the exact minimum value the symbol will take. This is more precise
     * than lower bound constraints.
     */
    void tight_lower_bound(const Expression tight_lb);

    /**
     * @brief Gets the tight upper bound
     * @return The exact maximum value the symbol can take, or null if unknown
     *
     * Unlike upper_bounds which are constraints, the tight upper bound represents
     * the exact maximum value the symbol will take during execution.
     */
    const Expression tight_upper_bound() const;

    /**
     * @brief Sets the tight upper bound
     * @param tight_ub The exact maximum value
     *
     * Sets the exact maximum value the symbol will take. This is more precise
     * than upper bound constraints.
     */
    void tight_upper_bound(const Expression tight_ub);

    /**
     * @brief Checks if the symbol is constant
     * @return true if the symbol's value does not change
     *
     * A constant symbol has the same value throughout its scope. Non-constant
     * symbols may evolve (e.g., loop iteration variables).
     */
    bool constant() const;

    /**
     * @brief Sets whether the symbol is constant
     * @param constant true if the symbol is constant, false if it evolves
     */
    void constant(bool constant);

    /**
     * @brief Gets the evolution map for this symbol
     * @return Expression describing how the symbol evolves, or null if none
     *
     * The map describes how the symbol's value changes, typically in loop iterations.
     * For example, a loop counter might have map(i) = i + 1, indicating it increments
     * by 1 each iteration.
     */
    const Expression map() const;

    /**
     * @brief Sets the evolution map for this symbol
     * @param map Expression describing how the symbol evolves
     *
     * Sets how the symbol evolves over time or iterations. This is used to track
     * induction variables in loops and analyze their behavior.
     */
    void map(const Expression map);

    /**
     * @brief Creates an assumption with bounds derived from a type
     * @param symbol The symbol to create assumption for
     * @param type The type to derive bounds from
     * @return Assumption with bounds matching the type's range
     *
     * Creates an assumption initialized with bounds appropriate for the given type.
     * For example, an Int32 type would set bounds to [-2^31, 2^31-1].
     *
     * @code
     * auto i = symbolic::symbol("i");
     * auto int32_type = types::Scalar::create(types::PrimitiveType::Int32);
     * auto assumption = Assumption::create(i, int32_type);
     * @endcode
     */
    static Assumption create(const Symbol symbol, const types::IType& type);
};

/**
 * @brief Map from symbols to their assumptions
 *
 * This type represents a collection of assumptions for multiple symbols,
 * typically used to describe the symbolic context for analyzing an SDFG region.
 * Each symbol maps to its Assumption containing bounds, constness, and evolution.
 */
typedef std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> Assumptions;

} // namespace symbolic
} // namespace sdfg
