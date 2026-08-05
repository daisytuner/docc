#include "sdfg/visitor/structured_sdfg_walker.h"
#include <stdexcept>
#include "sdfg/exceptions.h"


namespace sdfg::visitor {


StructuredSDFGWalker::Iterator::Iterator(ControlFlowNode* node, int32_t idx) : node_(node), idx_(idx) {}

std::pair<ControlFlowNode&, StructuredSDFGWalker::Scope> StructuredSDFGWalker::Iterator::operator*() const {
    auto scoped = (idx_ & SCOPE_MASK);
    if (scoped & SCOPE_ANY_MASK) {
        if ((scoped & IFELSE_ENTER) == IFELSE_ENTER) {
            return {*node_, Scope::IF_ENTRY};
        } else if ((scoped & IFELSE_EXIT) == IFELSE_EXIT) {
            return {*node_, Scope::IF_EXIT};
        } else if (scoped & SCOPE_ENTRY_FLAG) {
            return {*node_, Scope::ENTRY};
        } else if (scoped & SCOPE_EXIT_FLAG) {
            return {*node_, Scope::EXIT};
        } else {
            throw std::runtime_error("Invalid scope flags on #" + std::to_string(node_->element_id()));
        }
    } else if (auto* ifelse = dyn_cast<IfElse*>(node_)) {
        return {*ifelse, Scope::IF_NEXT_BRANCH};
    } else {
        return {*node_, Scope::NONE};
    }
}

bool StructuredSDFGWalker::Iterator::operator!=(const Iterator& other) const {
    return node_ != other.node_ || idx_ != other.idx_;
}

StructuredSDFGWalker::Iterator& StructuredSDFGWalker::Iterator::operator++() { return next_internal(true); }

StructuredSDFGWalker::Iterator& StructuredSDFGWalker::Iterator::next() { return next_internal(true); }

StructuredSDFGWalker::Iterator& StructuredSDFGWalker::Iterator::next_no_descend() { return next_internal(false); }

StructuredSDFGWalker::Iterator& StructuredSDFGWalker::Iterator::next_internal(bool descend) {
    auto* node = node_;

    if (node == nullptr) {
        throw std::out_of_range("Reached root");
    } else if (auto* seq = dyn_cast<Sequence*>(node)) {
        auto idx = idx_ & SEQ_MASK;
        if (idx & SCOPE_ENTRY_FLAG) {
            if (descend) {
                if (seq->size() > 0) {
                    parent_cache_.push_back({node, 0});
                    node_ = &seq->at(0);
                    set_enter_node_idx(*node_, descend);
                } else {
                    idx_ = SEQ_EXIT;
                }
            } else {
                move_to_parent_next(seq, descend);
            }
        } else if (idx & SCOPE_EXIT_FLAG) {
            move_to_parent_next(seq, descend);
        } else {
            throw std::runtime_error("Invalid Sequence scope flags on #" + std::to_string(node->element_id()));
        }
    } else if (auto* block = dyn_cast<Block*>(node)) {
        move_to_parent_next(block, descend);
    } else if (auto* ablock = dyn_cast<AssignmentBlock*>(node)) {
        move_to_parent_next(ablock, descend);
    } else if (auto* if_else = dyn_cast<IfElse*>(node)) {
        auto idx = idx_;
        auto flags = idx & IFELSE_MASK;
        if ((flags & IFELSE_FLAG) == IFELSE_FLAG) {
            if (flags & SCOPE_ENTRY_FLAG) {
                if (descend) {
                    if (if_else->size() > 0) {
                        parent_cache_.push_back({node, 0});
                        node_ = &if_else->at(0).first;
                        set_enter_node_idx(*node_, descend);
                    } else {
                        idx_ = IFELSE_EXIT;
                    }
                } else {
                    move_to_parent_next(if_else, descend);
                }
            } else if (flags & SCOPE_EXIT_FLAG) {
                move_to_parent_next(if_else, descend);
            } else {
                throw std::runtime_error("Invalid IfElse scope flags on #" + std::to_string(node->element_id()));
            }
        } else {
            if (idx >= 0) {
                if (descend) {
                    if (if_else->size() > idx) {
                        parent_cache_.push_back({node, idx});
                        node_ = &if_else->at(idx).first;
                        set_enter_node_idx(*node_, descend);
                    } else {
                        throw std::out_of_range(
                            "IfElse #" + std::to_string(node->element_id()) + " does not have entry " +
                            std::to_string(idx) + " (size " + std::to_string(if_else->size()) + ")"
                        );
                    }
                } else {
                    move_to_parent_next(if_else, descend);
                }
            } else {
                throw std::runtime_error("Invalid IfElse scope flags on #" + std::to_string(node->element_id()));
            }
        }
    } else if (auto* while_stmt = dyn_cast<While*>(node)) {
        auto idx = idx_ & LOOP_MASK;
        if (idx & SCOPE_ENTRY_FLAG) {
            if (descend) {
                parent_cache_.push_back({node, 0});
                node_ = &while_stmt->root();
                set_enter_node_idx(*node_, descend);
            } else {
                move_to_parent_next(while_stmt, descend);
            }
        } else if (idx & SCOPE_EXIT_FLAG) {
            move_to_parent_next(while_stmt, descend);
        } else {
            throw std::runtime_error("Invalid Loop scope flags on #" + std::to_string(node->element_id()));
        }
    } else if (auto* loop = dyn_cast<StructuredLoop*>(node)) {
        auto idx = idx_ & LOOP_MASK;
        if (idx & SCOPE_ENTRY_FLAG) {
            if (descend) {
                parent_cache_.push_back({node, 0});
                node_ = &loop->root();
                set_enter_node_idx(*node_, descend);
            } else {
                move_to_parent_next(loop, descend);
            }
        } else if (idx & SCOPE_EXIT_FLAG) {
            move_to_parent_next(loop, descend);
        } else {
            throw std::runtime_error("Invalid Loop scope flags on #" + std::to_string(node->element_id()));
        }
    } else if (auto* return_stmt = dyn_cast<Return*>(node)) {
        move_to_parent_next(return_stmt, descend);
    } else if (auto* break_stmt = dyn_cast<Break*>(node)) {
        move_to_parent_next(break_stmt, descend);
    } else if (auto* continue_stmt = dyn_cast<Continue*>(node)) {
        move_to_parent_next(continue_stmt, descend);
    } else {
        throw std::runtime_error("Unsupported control flow node type on #" + std::to_string(node->element_id()));
    }
    return *this;
}

void StructuredSDFGWalker::Iterator::move_to_parent_next(ControlFlowNode* child, bool descend) {
    if (parent_cache_.empty()) {
        auto parent = child->get_parent();
        if (parent == nullptr) {
            node_ = nullptr;
            idx_ = 0;
        } else {
            if (auto* seq = dyn_cast<Sequence*>(parent)) {
                auto idx = seq->index(*child);
                if (idx < 0) {
                    throw InvalidSDFGException("Child node not found in parent sequence");
                }
                move_to_next_seq_child(seq, idx, descend, nullptr);
            } else if (auto* if_else = dyn_cast<IfElse*>(parent)) {
                auto* branch = dyn_cast<Sequence*>(child);
                if (branch == nullptr) {
                    throw InvalidSDFGException(
                        "#" + std::to_string(child->element_id()) +
                        " has IfElse as parent, but is not a Sequence branch"
                    );
                }
                auto child_idx = if_else->index(*branch);
                if (child_idx < 0) {
                    throw InvalidSDFGException(
                        "#" + std::to_string(child->element_id()) + " has IfElse as parent, but is not a branch of it"
                    );
                }
                move_to_next_ifelse_child(if_else, child_idx, descend, nullptr);
            } else if (auto* while_stmt = dyn_cast<While*>(parent)) {
                move_to_next_loop_child(while_stmt, nullptr);
            } else if (auto* loop = dyn_cast<StructuredLoop*>(parent)) {
                move_to_next_loop_child(loop, nullptr);
            } else {
                throw std::runtime_error("Unsupported control flow node type on #" + std::to_string(parent->element_id()));
            }
        }
    } else {
        auto& parent_loc = parent_cache_.back();
        auto* parent = parent_loc.node_;
        auto prev_idx = parent_loc.idx_;

        move_to_next_child(parent, prev_idx, descend, &parent_loc);
    }
}

void StructuredSDFGWalker::Iterator::move_to_next_loop_child(ControlFlowNode* parent, ParentScopeLoc* parent_cache) {
    node_ = parent;
    idx_ = LOOP_EXIT;
    if (parent_cache) {
        parent_cache_.pop_back();
    }
}

void StructuredSDFGWalker::Iterator::set_enter_node_idx(ControlFlowNode& child, bool descend) {
    if (auto* seq = dyn_cast<Sequence*>(&child)) {
        idx_ = SEQ_ENTER;
    } else if (auto* if_else = dyn_cast<IfElse*>(&child)) {
        idx_ = IFELSE_ENTER;
    } else if (auto* while_stmt = dyn_cast<While*>(&child)) {
        idx_ = LOOP_ENTER;
    } else if (auto* loop = dyn_cast<StructuredLoop*>(&child)) {
        idx_ = LOOP_ENTER;
    } else if (is_a(
                   child.type_id(),
                   ElementType::AssignmentBlock | ElementType::Block | ElementType::Continue | ElementType::Break |
                       ElementType::Return
               )) {
        idx_ = NO_SCOPE_FLAG;
    } else {
        throw std::runtime_error("Unsupported control flow node type on #" + std::to_string(child.element_id()));
    }
}

void StructuredSDFGWalker::Iterator::
    move_to_next_seq_child(Sequence* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache) {
    auto next_idx = prev_idx + 1;
    auto size = parent->size();
    if (next_idx > size) {
        throw std::out_of_range(
            "Sequence #" + std::to_string(parent->element_id()) + " does not have entry " + std::to_string(next_idx) +
            " (size " + std::to_string(size) + ")"
        );
    } else if (next_idx == size) { // no more children, exit sequence
        node_ = parent;
        idx_ = SEQ_EXIT;
        if (parent_cache) {
            parent_cache_.pop_back();
        }
    } else { // next child exists
        if (parent_cache) {
            parent_cache->idx_ = next_idx;
        } else {
            parent_cache_.push_back({parent, next_idx});
        }
        auto& next_child = parent->at(next_idx);
        node_ = &next_child;
        set_enter_node_idx(next_child, descend);
    }
}

void StructuredSDFGWalker::Iterator::
    move_to_next_ifelse_child(IfElse* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache) {
    auto next_idx = prev_idx + 1;
    auto size = parent->size();
    if (next_idx > size) {
        throw std::out_of_range(
            "IfElse #" + std::to_string(parent->element_id()) + " does not have entry " + std::to_string(next_idx) +
            " (size " + std::to_string(size) + ")"
        );
    } else if (next_idx == size) { // no more children, exit sequence
        node_ = parent;
        idx_ = IFELSE_EXIT;
        if (parent_cache) {
            parent_cache_.pop_back();
        }
    } else { // next child exists
        node_ = parent;
        idx_ = next_idx;
        // go through IFELSE Branch marker, so that users understand that the next child won't be a successor in
        // execution order!
        if (parent_cache) {
            parent_cache_.pop_back();
        }
    }
}

void StructuredSDFGWalker::Iterator::
    move_to_next_child(ControlFlowNode* parent, int32_t prev_idx, bool descend, ParentScopeLoc* parent_cache) {
    if (auto* seq = dyn_cast<Sequence*>(parent)) {
        move_to_next_seq_child(seq, prev_idx, descend, parent_cache);
    } else if (auto* if_else = dyn_cast<IfElse*>(parent)) {
        move_to_next_ifelse_child(if_else, prev_idx, descend, parent_cache);
    } else if (auto* while_stmt = dyn_cast<While*>(parent)) {
        move_to_next_loop_child(while_stmt, parent_cache);
    } else if (auto* loop = dyn_cast<StructuredLoop*>(parent)) {
        move_to_next_loop_child(loop, parent_cache);
    } else {
        throw std::runtime_error("Unsupported control flow node type on #" + std::to_string(parent->element_id()));
    }
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::root(StructuredSDFG& sdfg) {
    auto& root_node = sdfg.root();
    return Iterator(&root_node, SEQ_ENTER); // start at entry of root sequence
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::from_node(ControlFlowNode& node) {
    if (auto* seq = dyn_cast<Sequence*>(&node)) {
        return Iterator(&node, SEQ_ENTER); // start at entry of sequence
    } else if (auto* if_else = dyn_cast<IfElse*>(&node)) {
        return Iterator(&node, IFELSE_ENTER); // start at entry of if-else
    } else if (auto* while_stmt = dyn_cast<While*>(&node)) {
        return Iterator(&node, LOOP_ENTER); // start at entry of while loop
    } else if (auto* loop = dyn_cast<StructuredLoop*>(&node)) {
        return Iterator(&node, LOOP_ENTER); // start at entry of structured loop
    } else {
        return Iterator(&node, NO_SCOPE_FLAG); // simple Leaf node, no scope
    }
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::end() {
    return Iterator(nullptr, 0); // end iterator
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::from_after(ControlFlowNode& node) {
    auto it = from_node(node);
    ++it;
    return std::move(it);
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::sequence_exit(Sequence& seq) {
    return Iterator(&seq, SEQ_EXIT); // exit of sequence
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::ifelse_exit(IfElse& ifelse) {
    return Iterator(&ifelse, IFELSE_EXIT); // exit of if-else
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::loop_exit(While& loop) {
    return Iterator(&loop, LOOP_EXIT); // exit of while loop
}

StructuredSDFGWalker::Iterator StructuredSDFGWalker::loop_exit(StructuredLoop& loop) {
    return Iterator(&loop, LOOP_EXIT); // exit of structured loop
}


} // namespace sdfg::visitor
