#pragma once

#include <boost/lexical_cast.hpp>
#include <memory>
#include <string>

#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief Base class for all structured control flow nodes in a StructuredSDFG
 *
 * ControlFlowNode is the abstract base class for all elements that form the structured
 * control flow of a StructuredSDFG. This includes:
 * - Block: A state containing a dataflow graph
 * - Sequence: A sequential container of control flow nodes
 * - IfElse: Conditional branching with multiple cases
 * - StructuredLoop: Base for structured loops (For, Map, While)
 * - Return: Function return statement
 * - Break/Continue: Loop control statements
 *
 * Control flow nodes form a hierarchical tree structure representing the program's
 * control flow in a structured manner, avoiding the need for arbitrary goto statements
 * or unstructured control flow graphs.
 *
 * @see Block
 * @see Sequence
 * @see IfElse
 * @see StructuredLoop
 * @see Return
 */
class ControlFlowNode : public Element {
    friend class builder::StructuredSDFGBuilder;

protected:
    ControlFlowNode* parent_;

    /**
     * @brief Protected constructor for control flow nodes with parent
     * @param element_id Unique identifier for this element
     * @param debug_info Debug information for this node
     * @param parent The parent ControlFlowNode
     */
    ControlFlowNode(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

public:
    /// Bitmask of all concrete ElementTypes that are ControlFlowNodes.
    static constexpr ElementType TypeGroup = ElementType::Block | ElementType::Sequence | ElementType::IfElse |
                                             ElementType::While | ElementType::Continue | ElementType::Break |
                                             ElementType::Return | ElementType::For | ElementType::Map |
                                             ElementType::Reduce;

    /// LLVM-style RTTI predicate: true if \p element is a ControlFlowNode.
    static bool classof(const Element& element) { return is_a(element.type_id(), TypeGroup); }

    virtual ~ControlFlowNode() = default;

    ControlFlowNode* get_parent();
    const ControlFlowNode* get_parent() const;

    ControlFlowNode(const ControlFlowNode& node) = delete;
    ControlFlowNode& operator=(const ControlFlowNode&) = delete;

    static std::vector<ControlFlowNode*> parent_chain(ControlFlowNode& child);
};

} // namespace structured_control_flow
} // namespace sdfg
