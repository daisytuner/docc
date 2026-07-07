#pragma once

#include <memory>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief Represents a conditional if-else control flow node with multiple cases
 *
 * The IfElse node represents structured conditional branching in a StructuredSDFG.
 * It consists of one or more cases, each with:
 * - A condition (boolean expression) that determines when the case executes
 * - A sequence (control flow body) that executes when the condition is true
 *
 * Cases are evaluated in order. At runtime, the first case whose condition evaluates
 * to true will be executed. This is similar to if-else-if chains in programming languages.
 *
 * **Completeness:**
 *
 * An IfElse is considered "complete" if its conditions exhaustively cover all possible
 * cases. This means that at least one condition will always be true at runtime. The
 * is_complete() method uses conjunctive normal form (CNF) analysis to determine if
 * the disjunction (OR) of all conditions is a tautology (always true).
 *
 * Complete examples:
 * - `if (x > 0) ... else if (x <= 0) ...` - covers all values of x
 * - `if (condition) ... else if (!condition) ...` - complementary conditions
 * - `if (true) ...` - always executes
 *
 * Incomplete examples:
 * - `if (x > 0) ...` - doesn't handle x <= 0
 * - `if (x > 10) ... else if (x < 5) ...` - gap between 5 and 10
 *
 * @see is_complete()
 * @see Sequence
 */
class IfElse : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::vector<std::unique_ptr<Sequence>> cases_;
    std::vector<symbolic::Condition> conditions_;

    IfElse(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    IfElse(const IfElse& node) = delete;
    IfElse& operator=(const IfElse&) = delete;

    ElementType type_id() const override { return ElementType::IfElse; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::IfElse; }

    void validate(const Function& function) const override;

    /**
     * @brief Returns the number of cases (if/else-if branches) in this IfElse
     * @return Number of cases
     */
    size_t size() const;

    /**
     * @brief Access a case by index (const version)
     * @param i Index of the case to access (0-based)
     * @return Pair of (sequence, condition) for the case at index i
     * @throws std::out_of_range if i >= size()
     */
    std::pair<const Sequence&, const symbolic::Condition> at(size_t i) const;

    /**
     * @brief Access a case by index (non-const version)
     * @param i Index of the case to access (0-based)
     * @return Pair of (sequence, condition) for the case at index i
     * @throws std::out_of_range if i >= size()
     */
    std::pair<Sequence&, const symbolic::Condition> at(size_t i);

    /**
     * @brief Checks if this IfElse is complete (all cases are covered)
     *
     * An IfElse is complete if the disjunction (OR) of all its conditions is a tautology,
     * meaning at least one condition will always be true regardless of variable values.
     *
     * This method uses conjunctive normal form (CNF) conversion to analyze whether the
     * combined conditions form a tautology. Each clause in the CNF must be a tautology
     * for the entire expression to be one.
     *
     * **Examples of complete IfElse nodes:**
     * - `if (x > 0) { ... } else if (x <= 0) { ... }` - True, covers all x
     * - `if (x > 0 || y > 0) { ... } else if (x <= 0 && y <= 0) { ... }` - True
     * - `if (true) { ... }` - True, always executes
     *
     * **Examples of incomplete IfElse nodes:**
     * - `if (x > 0) { ... }` - False, missing x <= 0 case
     * - `if (x > 10) { ... } else if (x < 5) { ... }` - False, gap for 5 <= x <= 10
     * - `if (x > 0 && y > 0) { ... } else if (x < 0 && y < 0) { ... }` - False, missing mixed signs
     *
     * @return true if all possible cases are covered, false otherwise
     */
    bool is_complete() const;

    /**
     * @brief Replace occurrences of an expression in all conditions and sequences
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

} // namespace structured_control_flow
} // namespace sdfg
