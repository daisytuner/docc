#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::visitor {

/**
 * @brief A sequential walker for traversing structured control flow nodes in a StructuredSDFG with iterators
 * Iterators traverse the SDFG same as our visitors do.
 * With elements that have children being visited
 *  * on entering
 *  * then all the children
 *  * again on exiting
 * Loops always have exactly 1 child: their root sequence.
 * Empty sequences will still be visited twice, as Scope::ENTRY and as Scope::EXIT
 * We can also move to the next element without entering children. In this case scopes will be visited as before, with
 * Scope::ENTRY. But calling `next_no_descend()` on a Scope::ENTRY (Sequence, IfElse, While, StructuredLoop) will leave
 * that node entirely, skipping over any potential children and the Scope::EXIT. If a scope was entered before, using
 * no_descend has no effect. It will step through all the children and Scope::EXIT
 *
 * Example SDFG { Map0 { Block0 } } from root:
 *  1. Root Seq Scope::ENTRY
 *  2. Map0 Scope::ENTRY
 *  3. Map0.root-Seq Scope::ENTRY
 *  4. Block0 Scope::NONE
 *  5. Map0.root-Seq Scope::EXIT
 *  6. Map0 Scope::EXIT
 *  7. Root Seq Scope::EXIT
 *
 * Example from `SequentialWalker::from_node(Map0.root-seq)` != SequentialWalker::loop_exit(Map0) with no_descend:
 *  1. Map0.root-seq Scope::ENTRY
 *  2. Map0 Scope::EXIT
 *
 *  IfElse has special handling, as its children are not in execution order. To this end, the walker will use
 * Scope::IF_ENTRY and Scope::IF_EXIT to signify that. It will additionally visit the IfElse node itself before every
 * branch (other than the first one). Example:
 *   1. IfElse Scope::IF_ENTRY
 *   2. IfElse.branch0.root Scope::ENTRY
 *   3. IfElse.branch0.root Scope::EXIT
 *   4. IfElse Scope::IF_NEXT_BRANCH
 *   5. IfElse.branch1.root Scope::ENTRY
 *   ...
 *   7. IfElse.branch1.root Scope::EXIT
 *   8. IfElse Scope::IF_EXIT
 */
class StructuredSDFGWalker {
public:
    enum class Scope { NONE, ENTRY, EXIT, IF_ENTRY, IF_EXIT, IF_NEXT_BRANCH };

    static constexpr std::string_view scope_to_string(Scope scope) {
        switch (scope) {
            case Scope::NONE:
                return "NONE";
            case Scope::ENTRY:
                return "ENTRY";
            case Scope::EXIT:
                return "EXIT";
            case Scope::IF_ENTRY:
                return "IF_ENTRY";
            case Scope::IF_EXIT:
                return "IF_EXIT";
            case Scope::IF_NEXT_BRANCH:
                return "IF_NEXT_BRANCH";
        }
        return "UNKNOWN";
    }

    class Iterator;

    class ParentScopeLoc {
        friend StructuredSDFGWalker;
        friend Iterator;

        ControlFlowNode* node_;
        int32_t idx_;
        ParentScopeLoc(ControlFlowNode* node, int32_t idx) : node_(node), idx_(idx) {}
    };

    class Iterator {
        friend StructuredSDFGWalker;

    private:
        ControlFlowNode* node_;
        int32_t idx_;
        std::list<ParentScopeLoc> parent_cache_;

        Iterator(ControlFlowNode* node, int32_t idx);

        Iterator& next_internal(bool descend);

        void move_to_parent_next(ControlFlowNode* child, bool descend);

        //
        void move_to_next_child(ControlFlowNode* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache);
        void move_to_next_seq_child(Sequence* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache);
        void move_to_next_ifelse_child(IfElse* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache);
        void move_to_next_loop_child(ControlFlowNode* parent, ParentScopeLoc* parent_cache);

        void set_enter_node_idx(ControlFlowNode& child, bool descend);

    public:
        bool operator!=(const Iterator& other) const;
        Iterator& operator++();

        /**
         * Move to next entry in exec order
         */
        Iterator& next();

        /**
         * Move to next entry in exec order, but do not descend into scopes, just visit scopes as Leafs.
         */
        Iterator& next_no_descend();
        std::pair<ControlFlowNode&, Scope> operator*() const;
    };

private:
    static constexpr int32_t NO_SCOPE_FLAG = 0x80000000;
    static constexpr int32_t SCOPE_MASK = 0x8000007F;
    static constexpr int32_t SCOPE_ANY_MASK = 0x70;
    static constexpr int32_t SCOPE_ENTRY_FLAG = 0x00000001;
    static constexpr int32_t SCOPE_EXIT_FLAG = 0x00000002;
    static constexpr int32_t SEQ_FLAG = 0x80000010;
    static constexpr int32_t SEQ_MASK = 0x8000001F;
    static constexpr int32_t SEQ_ENTER = SEQ_FLAG | SCOPE_ENTRY_FLAG;
    static constexpr int32_t SEQ_EXIT = SEQ_FLAG | SCOPE_EXIT_FLAG;
    static constexpr int32_t LOOP_FLAG = 0x80000020;
    static constexpr int32_t LOOP_MASK = 0x8000002F;
    static constexpr int32_t LOOP_ENTER = LOOP_FLAG | SCOPE_ENTRY_FLAG;
    static constexpr int32_t LOOP_EXIT = LOOP_FLAG | SCOPE_EXIT_FLAG;
    static constexpr int32_t IFELSE_FLAG = 0x80000040;
    static constexpr int32_t IFELSE_MASK = 0x8000004F;
    static constexpr int32_t IFELSE_ENTER = IFELSE_FLAG | SCOPE_ENTRY_FLAG;
    static constexpr int32_t IFELSE_EXIT = IFELSE_FLAG | SCOPE_EXIT_FLAG;

public:
    static Iterator root(StructuredSDFG& sdfg);
    static Iterator from_node(ControlFlowNode& node);
    static Iterator sequence_exit(Sequence& seq);
    static Iterator ifelse_exit(IfElse& ifelse);
    static Iterator loop_exit(While& loop);
    static Iterator loop_exit(StructuredLoop& loop);
    static Iterator end();

    static Iterator from_after(ControlFlowNode& node);

protected:
};

inline std::ostream& operator<<(std::ostream& os, StructuredSDFGWalker::Scope scope) {
    os << StructuredSDFGWalker::scope_to_string(scope);
    return os;
}

} // namespace sdfg::visitor
