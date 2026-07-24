#pragma once

#include <boost/lexical_cast.hpp>
#include <nlohmann/json.hpp>
#include <string>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
}

namespace control_flow {

/**
 * @brief A state in a Stateful DataFlow Graph (SDFG).
 *
 * States are the fundamental building blocks of an SDFG control-flow graph. Each state
 * represents a computational stage and contains a data-flow graph describing the
 * computations performed when the state is executed.
 *
 * SDFGs can be constructed in two ways:
 * 1. **Control-Flow Approach**: Using State nodes connected by InterstateEdge edges
 *    (this is the approach using this class). This creates a general directed graph
 *    that can contain cycles.
 * 2. **Structured Control-Flow Approach**: Using structured_control_flow elements
 *    (e.g., For, While, IfElse) which yield an acyclic StructuredSDFG.
 *
 * @see InterstateEdge for transitions between states
 * @see data_flow::DataFlowGraph for the computational graph within a state
 */
class State : public Element {
    friend class sdfg::builder::SDFGBuilder;

private:
    // Remark: Exclusive resource
    const graph::Vertex vertex_;
    std::unique_ptr<data_flow::DataFlowGraph> dataflow_;

protected:
    State(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex);

public:
    // Remark: Exclusive resource
    State(const State& state) = delete;
    State& operator=(const State&) = delete;

    ElementType type_id() const override { return ElementType::State; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::State; }

    /**
     * @brief Validates the state within the context of a function.
     * @param function The function containing this state
     * @throws InvalidSDFGException if validation fails
     */
    void validate(const Function& function) const override;

    /**
     * @brief Returns the graph vertex representing this state.
     * @return The graph vertex associated with this state
     */
    graph::Vertex vertex() const;

    /**
     * @brief Returns the data-flow graph of this state (const version).
     * @return Const reference to the data-flow graph containing computational nodes
     */
    const data_flow::DataFlowGraph& dataflow() const;

    /**
     * @brief Returns the data-flow graph of this state (mutable version).
     * @return Mutable reference to the data-flow graph containing computational nodes
     */
    data_flow::DataFlowGraph& dataflow();

    /**
     * @brief Replaces symbolic expressions in the state's data-flow graph.
     * @param old_expression The expression to replace
     * @param new_expression The replacement expression
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

/**
 * @brief A special state that returns a value from the SDFG.
 *
 * ReturnState represents the termination point of an SDFG execution path.
 * It can return either:
 * - A data container (variable) defined in the function
 * - A constant value with an associated type
 *
 * A ReturnState must not have any outgoing edges.
 *
 * @see State for the base state functionality
 */
class ReturnState : public State {
    friend class sdfg::builder::SDFGBuilder;

private:
    std::string data_;
    std::unique_ptr<types::IType> type_;

    ReturnState(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, const std::string& data);

    ReturnState(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        const std::string& data,
        const types::IType& type
    );

public:
    // Remark: Exclusive resource
    ReturnState(const ReturnState& state) = delete;
    ReturnState& operator=(const ReturnState&) = delete;

    /**
     * @brief Returns the data or constant value being returned.
     * @return The name of the data container or the constant value string
     */
    const std::string& data() const;

    /**
     * @brief Returns the type of the constant being returned.
     * @return The type of the constant (only valid if is_constant() returns true)
     */
    const types::IType& type() const;

    /**
     * @brief Checks if this return state returns a data container.
     * @return true if returning a data container, false otherwise
     */
    bool is_data() const;

    /**
     * @brief Checks if this return state returns a constant value.
     * @return true if returning a constant value, false otherwise
     */
    bool is_constant() const;

    /**
     * @brief Validates the return state within the context of a function.
     * @param function The function containing this return state
     * @throws InvalidSDFGException if validation fails (e.g., has outgoing edges)
     */
    void validate(const Function& function) const override;

    /**
     * @brief Replaces symbolic expressions in the return state.
     * @param old_expression The expression to replace
     * @param new_expression The replacement expression
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace control_flow
} // namespace sdfg
