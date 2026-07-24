#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief Represents a while-loop with condition-controlled iteration
 *
 * A While loop repeatedly executes its body sequence while a condition remains true.
 * Unlike For loops, While loops don't have explicit initialization or update expressions.
 *
 * The loop body can contain any control flow nodes, including Break and Continue
 * statements to control loop execution.
 *
 * **Example:**
 * ```cpp
 * while (condition) {
 *   // body
 * }
 * ```
 *
 * @see For
 * @see Break
 * @see Continue
 */
class While : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::unique_ptr<Sequence> root_;

    While(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 2;

public:
    While(const While& node) = delete;
    While& operator=(const While&) = delete;

    ElementType type_id() const override { return ElementType::While; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::While; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    /**
     * @brief Access the loop body sequence (const version)
     * @return Const reference to the sequence containing the loop body
     */
    const Sequence& root() const;

    /**
     * @brief Access the loop body sequence (non-const version)
     * @return Reference to the sequence containing the loop body
     */
    Sequence& root();

    /**
     * @brief Replace occurrences of an expression in the loop body
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/**
 * @brief Represents a break statement that exits the innermost loop
 *
 * A Break statement immediately terminates the execution of the innermost
 * enclosing loop (For, While, or Map). Control flow continues after the loop.
 *
 * Break statements are typically used within conditional branches inside loops
 * to provide early exit conditions.
 */
class Break : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Break(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    ElementType type_id() const override { return ElementType::Break; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Break; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/**
 * @brief Represents a continue statement that skips to the next loop iteration
 *
 * A Continue statement immediately skips the rest of the current iteration of
 * the innermost enclosing loop (For, While, or Map) and begins the next iteration
 * (if the loop condition is still satisfied).
 *
 * Continue statements are typically used within conditional branches inside loops
 * to skip certain iterations based on runtime conditions.
 */
class Continue : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Continue(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    ElementType type_id() const override { return ElementType::Continue; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Continue; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

} // namespace structured_control_flow
} // namespace sdfg
