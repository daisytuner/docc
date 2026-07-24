#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief Represents a return statement that exits the function with a value
 *
 * A Return node represents a function return statement in a StructuredSDFG.
 * It can return either:
 * - A data container (variable): References a container name in the SDFG
 * - A constant value: A literal value with an explicit type
 *
 * The return type must match the function's declared return type. Functions
 * with non-void return types typically have Return nodes as the final statement
 * in control flow paths.
 *
 * **Examples:**
 * - `return x;` - Returns the value of container "x" (data return)
 * - `return 42;` - Returns constant integer 42 (constant return)
 * - `return 3.14;` - Returns constant float 3.14 (constant return)
 *
 * @see StructuredSDFG
 */
class Return : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::string data_;
    std::unique_ptr<types::IType> type_;

    Return(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent, const std::string& data);

    Return(
        size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        const std::string& constant,
        const types::IType& type
    );

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    Return(const Return& Return) = delete;
    Return& operator=(const Return&) = delete;

    ElementType type_id() const override { return ElementType::Return; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Return; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    /**
     * @brief Get the data or constant value being returned
     * @return Container name (for data returns) or constant value string (for constant returns)
     */
    const std::string& data() const;

    /**
     * @brief Get the return type
     * @return Reference to the type of the returned value
     */
    const types::IType& type() const;

    /**
     * @brief Check if this return statement returns a data container
     * @return true if returning a container, false if returning a constant
     */
    bool is_data() const;

    /**
     * @brief Check if this return statement returns a constant value
     * @return true if returning a constant, false if returning a container
     */
    bool is_constant() const;

    void validate(const Function& function) const override;

    /**
     * @brief Replace occurrences of an expression (no-op for return nodes)
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    using Element::replace;
};

} // namespace structured_control_flow
} // namespace sdfg
