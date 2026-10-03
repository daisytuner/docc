#include "sdfg/users/users.h"

#include <cassert>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/sets.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace users {

void Users::on_enter(structured_control_flow::ControlFlowNode& node) {
    this->traversal_stack_.push_back({this->current_owner_, this->order_.size()});
    this->current_owner_ = &node;
}

void Users::on_leave(structured_control_flow::ControlFlowNode& node) {
    auto [outer_owner, begin] = this->traversal_stack_.back();
    this->traversal_stack_.pop_back();
    this->current_owner_ = outer_owner;
    this->ranges_[&node] = {begin, this->order_.size()};
}

void Users::on_use(const std::string& container, Element* element, Use use, LoopPart part) {
    if (part == LoopPart::None) {
        this->add_user(std::make_unique<User>(container, element, use));
    } else {
        this->add_user(
            std::make_unique<ForUser>(
                container, element, use, part == LoopPart::Init, part == LoopPart::Condition, part == LoopPart::Update
            )
        );
    }
}

Users::Users(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()) {

      };

Users::Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node)
    : Analysis(sdfg), node_(node) {

      };

void Users::run(analysis::AnalysisManager& analysis_manager) {
    users_.clear();
    order_.clear();
    ranges_.clear();
    current_owner_ = nullptr;
    traversal_stack_.clear();
    may_return_.clear();
    loop_conditions_.clear();
    this->users_lookup_.clear();
    this->container_users_.clear();

    this->traverse(node_);

    for (auto& entry : this->users_) {
        const auto& container = entry->container();
        if (entry->use() == Use::NOP || container.empty()) {
            continue;
        }
        auto& tables = this->container_users_[container];
        tables.uses.push_back(entry.get());
        switch (entry->use()) {
            case Use::READ:
                tables.reads.push_back(entry.get());
                break;
            case Use::WRITE:
                tables.writes.push_back(entry.get());
                break;
            case Use::VIEW:
                tables.views.push_back(entry.get());
                break;
            case Use::MOVE:
                tables.moves.push_back(entry.get());
                break;
            default:
                break;
        }
    }
};

const Users::ContainerUsers& Users::container_users(const std::string& container) const {
    auto it = this->container_users_.find(container);
    if (it != this->container_users_.end()) {
        return it->second;
    }
    if (!this->sdfg_.exists(container)) {
        throw std::out_of_range("Users: unknown container " + container);
    }
    static const ContainerUsers no_users;
    return no_users;
}

std::vector<User*> Users::uses() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->use() == Use::NOP) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

std::vector<User*> Users::uses(const std::string& container) const {
    if (!container.empty()) {
        auto it = this->container_users_.find(container);
        return it != this->container_users_.end() ? it->second.uses : std::vector<User*>{};
    }
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->container() != container) {
            continue;
        }
        if (entry->use() == Use::NOP) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

size_t Users::num_uses(const std::string& container) const {
    return this->uses(container).size();
};

std::vector<User*> Users::writes() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->use() != Use::WRITE) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

const std::vector<User*>& Users::writes(const std::string& container) const {
    return this->container_users(container).writes;
};

size_t Users::num_writes(const std::string& container) const {
    return this->writes(container).size();
};

std::vector<User*> Users::reads() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->use() != Use::READ) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

const std::vector<User*>& Users::reads(const std::string& container) const {
    return this->container_users(container).reads;
};

size_t Users::num_reads(const std::string& container) const {
    return this->reads(container).size();
};

std::vector<User*> Users::views() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->use() != Use::VIEW) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

const std::vector<User*>& Users::views(const std::string& container) const {
    return this->container_users(container).views;
};

size_t Users::num_views(const std::string& container) const {
    return this->views(container).size();
};

std::vector<User*> Users::moves() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry->use() != Use::MOVE) {
            continue;
        }
        us.push_back(entry.get());
    }

    return us;
};

const std::vector<User*>& Users::moves(const std::string& container) const {
    return this->container_users(container).moves;
};

size_t Users::num_moves(const std::string& container) const {
    return this->moves(container).size();
};

structured_control_flow::ControlFlowNode* Users::scope(User* user) {
    if (auto data_node = dynamic_cast<data_flow::DataFlowNode*>(user->element())) {
        return static_cast<structured_control_flow::Block*>(data_node->get_parent().get_parent());
    } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(user->element())) {
        return static_cast<structured_control_flow::Block*>(memlet->get_parent().get_parent());
    } else if (auto transition = dyn_cast<structured_control_flow::AssignmentBlock*>(user->element())) {
        return transition->get_parent();
    } else {
        auto user_element = dyn_cast<structured_control_flow::ControlFlowNode*>(user->element());
        assert(user_element != nullptr && "Users::scope: User element is not a ControlFlowNode");
        return user_element;
    }
}

// Containers in program order of their first use.
const std::vector<std::string> Users::all_containers_in_order() {
    std::unordered_set<std::string> unique_containers;
    std::vector<std::string> containers;
    for (auto* user : this->order_) {
        if (user->use() != Use::NOP && unique_containers.insert(user->container()).second) {
            containers.push_back(user->container());
        }
    }
    return containers;
}

UsersView::UsersView(Users& users, const structured_control_flow::ControlFlowNode& node) : users_(users), node_(&node) {
    std::tie(this->begin_, this->end_) = users.ranges_.at(&node);
};

bool UsersView::contains(const User& user) const {
    return user.index_ >= this->begin_ && user.index_ < this->end_;
}

std::vector<User*> UsersView::uses() const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() == Use::NOP) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::uses(const std::string& container) const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->container() != container) {
            continue;
        }
        if (user->use() == Use::NOP) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::writes() const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() != Use::WRITE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::writes(const std::string& container) const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::WRITE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::reads() const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() != Use::READ) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::reads(const std::string& container) const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::READ) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::views() const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() != Use::VIEW) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::views(const std::string& container) const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::VIEW) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::moves() const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() != Use::MOVE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::moves(const std::string& container) const {
    std::vector<User*> us;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::MOVE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::unordered_set<User*> UsersView::all_uses_after(User& user) {
    assert(this->contains(user));

    std::vector<Users::Interval> intervals;
    this->users_.forward_intervals(user, *this->node_, true, intervals);
    return this->users_.collect_uses(intervals, &user, nullptr, false);
};

Users::UserKey Users::user_key(Element* element, Use use, bool is_init, bool is_condition, bool is_update) {
    uint8_t loop_part = is_init ? 1 : is_condition ? 2 : is_update ? 3 : 0;
    return UserKey{element->element_id(), use, loop_part};
}

bool Users::
    has_user(const std::string& container, Element* element, Use use, bool is_init, bool is_condition, bool is_update) {
    auto it = this->users_lookup_.find(user_key(element, use, is_init, is_condition, is_update));
    if (it == this->users_lookup_.end()) {
        return false;
    }
    for (auto* user : it->second) {
        if (user->container() == container) {
            return true;
        }
    }
    return false;
}

User* Users::
    get_user(const std::string& container, Element* element, Use use, bool is_init, bool is_condition, bool is_update) {
    auto it = this->users_lookup_.find(user_key(element, use, is_init, is_condition, is_update));
    if (it != this->users_lookup_.end()) {
        for (auto* user : it->second) {
            if (user->container() == container) {
                return user;
            }
        }
    }
    throw std::out_of_range("Users: no user of " + container);
}

void Users::add_user(std::unique_ptr<User> user) {
    user->index_ = this->order_.size();
    user->owner_ = this->current_owner_;
    auto* user_ptr = user.get();
    this->users_.push_back(std::move(user));
    this->order_.push_back(user_ptr);
    bool is_init = false;
    bool is_condition = false;
    bool is_update = false;
    if (auto for_user = dynamic_cast<ForUser*>(user_ptr)) {
        auto for_loop = dyn_cast<structured_control_flow::StructuredLoop*>(user_ptr->element());
        if (for_loop == nullptr) {
            throw std::invalid_argument("Invalid user type");
        }
        if (for_user->is_init()) {
            is_init = true;
        } else if (for_user->is_condition()) {
            is_condition = true;
        } else if (for_user->is_update()) {
            is_update = true;
        } else {
            throw std::invalid_argument("Invalid user type");
        }
    }

    this->users_lookup_[user_key(user_ptr->element(), user_ptr->use(), is_init, is_condition, is_update)]
        .push_back(user_ptr);
}

std::unordered_set<std::string> Users::locals(structured_control_flow::ControlFlowNode& node) {
    auto& sdfg = this->sdfg_;

    // Locals have no uses outside of the node
    // We can check this by comparing the number of uses of the container in the view and the total
    // number of uses of the container in the users map.
    std::unordered_set<std::string> locals;
    UsersView view(*this, node);
    for (auto& user : view.uses()) {
        if (!sdfg.is_transient(user->container())) {
            continue;
        }
        if (view.uses(user->container()).size() == this->uses(user->container()).size()) {
            locals.insert(user->container());
        }
    }

    return locals;
};

// Containers in program order of their first use within the view.
const std::vector<std::string> UsersView::all_containers_in_order() {
    std::unordered_set<std::string> unique_containers;
    std::vector<std::string> containers;
    for (size_t k = this->begin_; k < this->end_; ++k) {
        auto* user = this->users_.order_[k];
        if (user->use() != Use::NOP && unique_containers.insert(user->container()).second) {
            containers.push_back(user->container());
        }
    }
    return containers;
}

/***** Structural dominance *****/
//
// Dominance and post-dominance between users are derived from the structured control flow instead
// of the user graph. Each rule mirrors the edges built by traverse_impl:
//  - Block / AssignmentBlock / IfElse condition reads / loop header and update users form chains
//    in creation (index) order.
//  - Sequence: children in order; a Return child is a dead end and its predecessor bypasses it.
//  - IfElse: condition chain -> every branch -> exit, plus condition chain -> exit if incomplete.
//  - While: s -> body -> t, s -> t, t -> s.
//  - StructuredLoop: header chain (reductions, init, indvar init, conditions) -> body -> update
//    chain -> t, every condition read -> t and back edge t -> last header user.

bool Users::traversed(const Node& node) const {
    return this->ranges_.find(&node) != this->ranges_.end();
}

const Users::Node* Users::child_towards(const Node& ancestor, const Node* node) const {
    if (node == &ancestor) {
        return nullptr;
    }
    while (node->get_parent() != &ancestor) {
        node = node->get_parent();
    }
    return node;
}

// A node is an ancestor (or owner) of a user iff the user's index lies in the node's range.
const Users::Node* Users::common_ancestor(const User& a, const User& b) const {
    for (const Node* n = a.owner_; n != nullptr; n = n->get_parent()) {
        auto it = this->ranges_.find(n);
        if (it == this->ranges_.end()) {
            return nullptr;
        }
        if (b.index_ >= it->second.first && b.index_ < it->second.second) {
            return n;
        }
    }
    return nullptr;
}

bool Users::may_return(const Node& node) {
    auto it = this->may_return_.find(&node);
    if (it != this->may_return_.end()) {
        return it->second;
    }

    bool result = false;
    if (dynamic_cast<const structured_control_flow::Return*>(&node)) {
        result = true;
    } else if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size() && !result; i++) {
            auto& child = seq->at(i);
            if (!this->traversed(child)) {
                break;
            }
            result = this->may_return(child);
        }
    } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size() && !result; i++) {
            result = this->may_return(if_else->at(i).first);
        }
    } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(&node)) {
        result = this->may_return(while_stmt->root());
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&node)) {
        result = this->may_return(loop->root());
    }

    this->may_return_[&node] = result;
    return result;
}

const std::vector<size_t>& Users::loop_conditions(const structured_control_flow::StructuredLoop& loop) {
    auto it = this->loop_conditions_.find(&loop);
    if (it != this->loop_conditions_.end()) {
        return it->second;
    }

    std::vector<size_t> conditions;
    auto [begin, end] = this->ranges_.at(&loop);
    auto body_begin = this->ranges_.at(&loop.root()).first;
    for (size_t k = begin; k < body_begin; k++) {
        auto* for_user = dynamic_cast<ForUser*>(this->order_[k]);
        if (for_user != nullptr && for_user->owner_ == &loop && for_user->is_condition()) {
            conditions.push_back(k);
        }
    }
    return this->loop_conditions_.emplace(&loop, std::move(conditions)).first->second;
}

// Does a dominate the exit of region (which contains a)?
bool Users::must_exit(const User& a, const Node& region) {
    if (a.owner_ == &region) {
        if (dynamic_cast<const structured_control_flow::Return*>(&region)) {
            return false;
        }
        if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
            auto& conditions = this->loop_conditions(*loop);
            return conditions.empty() || a.index_ <= conditions.front();
        }
        return true;
    }

    auto* child = this->child_towards(region, a.owner_);
    if (dynamic_cast<const structured_control_flow::Sequence*>(&region)) {
        return this->must_exit(a, *child);
    } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&region)) {
        return if_else->size() == 1 && if_else->is_complete() && this->must_exit(a, *child);
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
        return this->loop_conditions(*loop).empty() && this->must_exit(a, *child);
    }
    return false;
}

// Do all paths from b stay inside region until they reach its exit (no Return on the way)?
bool Users::no_escape(const User& b, const Node& region) {
    if (b.owner_ == &region) {
        if (dynamic_cast<const structured_control_flow::Return*>(&region)) {
            return false;
        }
        if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&region)) {
            return !this->may_return(*if_else);
        }
        if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
            return !this->may_return(loop->root());
        }
        return true;
    }

    auto* child = this->child_towards(region, b.owner_);
    if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(&region)) {
        if (!this->no_escape(b, *child)) {
            return false;
        }
        bool after = false;
        for (size_t i = 0; i < seq->size(); i++) {
            auto& sibling = seq->at(i);
            if (!this->traversed(sibling)) {
                break;
            }
            if (after && this->may_return(sibling)) {
                return false;
            }
            after = after || &sibling == child;
        }
        return true;
    } else if (dynamic_cast<const structured_control_flow::IfElse*>(&region)) {
        return this->no_escape(b, *child);
    } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(&region)) {
        return !this->may_return(while_stmt->root());
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
        return !this->may_return(loop->root());
    }
    return false;
}

// Does every path from the entry of region (which contains a) pass a before leaving the region?
bool Users::post_dominates_entry(const User& a, const Node& region) {
    if (a.owner_ == &region) {
        if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
            auto& conditions = this->loop_conditions(*loop);
            auto body_begin = this->ranges_.at(&loop->root()).first;
            if (a.index_ < body_begin) {
                return conditions.empty() || a.index_ <= conditions.front();
            }
            return conditions.empty() && !this->may_return(loop->root());
        }
        return true;
    }

    auto* child = this->child_towards(region, a.owner_);
    if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(&region)) {
        if (dynamic_cast<const structured_control_flow::Return*>(child)) {
            return false;
        }
        for (size_t i = 0; i < seq->size(); i++) {
            auto& sibling = seq->at(i);
            if (&sibling == child) {
                break;
            }
            if (this->may_return(sibling)) {
                return false;
            }
        }
        return this->post_dominates_entry(a, *child);
    } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&region)) {
        return if_else->size() == 1 && if_else->is_complete() && this->post_dominates_entry(a, *child);
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
        return this->loop_conditions(*loop).empty() && this->post_dominates_entry(a, *child);
    }
    return false;
}

bool Users::dominates(User& user1, User& user2) {
    if (&user1 == &user2 || user1.owner_ == nullptr || user2.owner_ == nullptr) {
        return false;
    }
    auto* lca = this->common_ancestor(user1, user2);
    if (lca == nullptr) {
        return false;
    }
    auto* c1 = this->child_towards(*lca, user1.owner_);
    auto* c2 = this->child_towards(*lca, user2.owner_);

    if (dynamic_cast<const structured_control_flow::Sequence*>(lca)) {
        return this->ranges_.at(c1).first < this->ranges_.at(c2).first && this->must_exit(user1, *c1);
    } else if (dynamic_cast<const structured_control_flow::IfElse*>(lca)) {
        if (c1 != nullptr) {
            return false;
        }
        return c2 != nullptr || user1.index_ < user2.index_;
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(lca)) {
        if (c1 == nullptr) {
            // The back edge enters the last condition read and every condition read may exit:
            // condition reads after the first and before the last only dominate their successors
            // up to the last condition read.
            auto& conditions = this->loop_conditions(*loop);
            if (!conditions.empty() && user1.index_ > conditions.front() && user1.index_ < conditions.back()) {
                return c2 == nullptr && user1.index_ < user2.index_ && user2.index_ < conditions.back();
            }
            return user1.index_ < user2.index_;
        }
        // user1 in body, user2 owned by the loop: only update users are dominated
        auto body_end = this->ranges_.at(&loop->root()).second;
        return user2.index_ >= body_end && this->must_exit(user1, *c1);
    }
    // Block, AssignmentBlock, Return: chains
    return user1.index_ < user2.index_;
}

bool Users::post_dominates(User& user1, User& user2) {
    if (&user1 == &user2 || user1.owner_ == nullptr || user2.owner_ == nullptr) {
        return false;
    }
    auto* lca = this->common_ancestor(user1, user2);
    if (lca == nullptr) {
        return false;
    }
    auto* c1 = this->child_towards(*lca, user1.owner_);
    auto* c2 = this->child_towards(*lca, user2.owner_);

    if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(lca)) {
        if (this->ranges_.at(c1).first < this->ranges_.at(c2).first) {
            return false;
        }
        // A Return is bypassed by its predecessor's edge to the sequence exit
        if (dynamic_cast<const structured_control_flow::Return*>(c1)) {
            return false;
        }
        if (!this->no_escape(user2, *c2)) {
            return false;
        }
        bool between = false;
        for (size_t i = 0; i < seq->size(); i++) {
            auto& sibling = seq->at(i);
            if (&sibling == c1) {
                break;
            }
            if (between && this->may_return(sibling)) {
                return false;
            }
            between = between || &sibling == c2;
        }
        return this->post_dominates_entry(user1, *c1);
    } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(lca)) {
        if (c1 == nullptr) {
            return c2 == nullptr && user1.index_ > user2.index_;
        }
        return c2 == nullptr && if_else->size() == 1 && if_else->is_complete() &&
               this->post_dominates_entry(user1, *c1);
    } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(lca)) {
        auto& conditions = this->loop_conditions(*loop);
        auto [body_begin, body_end] = this->ranges_.at(&loop->root());
        if (c1 == nullptr && c2 == nullptr) {
            if (user1.index_ < body_begin) {
                // Header user: reached from earlier header users before the first condition exit
                return user1.index_ > user2.index_ && (conditions.empty() || user1.index_ <= conditions.front());
            }
            if (user2.index_ >= body_end) {
                return user1.index_ > user2.index_;
            }
            return conditions.empty() && !this->may_return(loop->root());
        }
        if (c1 == nullptr) {
            // user1 owned, user2 in body: only update users post-dominate the body
            return user1.index_ >= body_end && this->no_escape(user2, *c2);
        }
        // user1 in body, user2 owned: only header users before a body that is always entered
        return user2.index_ < body_begin && conditions.empty() && this->post_dominates_entry(user1, *c1);
    }
    // Block, AssignmentBlock, Return: chains
    return user1.index_ > user2.index_;
}

/***** Structural reachability *****/

// The back edge of a loop enters its last header user (last condition read or the indvar init write).
size_t Users::loop_reentry(const structured_control_flow::StructuredLoop& loop) {
    auto begin = this->ranges_.at(&loop).first;
    for (size_t k = this->ranges_.at(&loop.root()).first; k-- > begin;) {
        auto* user = this->order_[k];
        if (user->owner_ == &loop && user->use() != Use::NOP) {
            return k;
        }
    }
    return begin;
}

// Users reachable from user without leaving stop; cut_at_stop drops stop's own back edge.
void Users::forward_intervals(const User& user, const Node& stop, bool cut_at_stop, std::vector<Interval>& out) {
    const Node* owner = user.owner_;
    if (dynamic_cast<const structured_control_flow::Return*>(owner)) {
        return;
    }
    auto loops_back = [&](const Node* node) {
        return node != &stop || !cut_at_stop;
    };

    auto owner_end = this->ranges_.at(owner).second;
    out.push_back({user.index_ + 1, owner_end});
    if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(owner); loop && loops_back(loop)) {
        out.push_back({this->loop_reentry(*loop), owner_end});
    }

    for (const Node* region = owner; region != &stop; region = region->get_parent()) {
        const Node* parent = region->get_parent();
        auto [parent_begin, parent_end] = this->ranges_.at(parent);
        auto region_end = this->ranges_.at(region).second;
        if (dynamic_cast<const structured_control_flow::Sequence*>(parent)) {
            out.push_back({region_end, parent_end});
        } else if (dynamic_cast<const structured_control_flow::While*>(parent)) {
            if (loops_back(parent)) {
                out.push_back({parent_begin, parent_end});
            }
        } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(parent)) {
            out.push_back({region_end, parent_end});
            if (loops_back(parent)) {
                out.push_back({this->loop_reentry(*loop), parent_end});
            }
        }
    }
}

// Users of region that reach its exit without passing user (user must dominate that exit).
void Users::exit_intervals(const User& user, const Node& region, std::vector<Interval>& out) {
    auto region_end = this->ranges_.at(&region).second;
    if (user.owner_ == &region) {
        out.push_back({user.index_ + 1, region_end});
        return;
    }
    auto* child = this->child_towards(region, user.owner_);
    this->exit_intervals(user, *child, out);
    if (dynamic_cast<const structured_control_flow::Sequence*>(&region) ||
        dynamic_cast<const structured_control_flow::StructuredLoop*>(&region)) {
        out.push_back({this->ranges_.at(child).second, region_end});
    }
}

// Users of region (which contains user) that reach user without leaving region.
void Users::entry_intervals(const User& user, const Node& region, std::vector<Interval>& out) {
    const Node* owner = user.owner_;
    auto [owner_begin, owner_end] = this->ranges_.at(owner);
    auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(owner);
    if (loop && user.index_ >= this->loop_reentry(*loop)) {
        out.push_back({owner_begin, owner_end});
    } else {
        out.push_back({owner_begin, user.index_});
    }

    for (const Node* child = owner; child != &region; child = child->get_parent()) {
        const Node* parent = child->get_parent();
        auto [parent_begin, parent_end] = this->ranges_.at(parent);
        if (dynamic_cast<const structured_control_flow::Sequence*>(parent)) {
            out.push_back({parent_begin, this->ranges_.at(child).first});
        } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(parent)) {
            out.push_back({parent_begin, this->ranges_.at(&if_else->at(0).first).first});
        } else {
            out.push_back({parent_begin, parent_end});
        }
    }
}

std::unordered_set<User*> Users::collect_uses(
    const std::vector<Interval>& intervals, const User* skip1, const User* skip2, bool skip_returns
) const {
    std::unordered_set<User*> uses;
    for (auto [begin, end] : intervals) {
        for (size_t k = begin; k < end; k++) {
            auto* user = this->order_[k];
            if (user == skip1 || user == skip2 || user->use() == Use::NOP) {
                continue;
            }
            // Return users are sinks and reach nothing
            if (skip_returns && dynamic_cast<const structured_control_flow::Return*>(user->owner_)) {
                continue;
            }
            uses.insert(user);
        }
    }
    return uses;
}

const std::unordered_set<User*> Users::all_uses_after(User& user) {
    std::vector<Interval> intervals;
    this->forward_intervals(user, this->node_, false, intervals);
    return this->collect_uses(intervals, &user, nullptr, false);
}

// Users on paths from user1 to user2 that do not pass user1 again. Requires that user1 dominates user2.
const std::unordered_set<User*> Users::all_uses_between(User& user1, User& user2) {
    std::vector<Interval> intervals;
    auto* lca = this->common_ancestor(user1, user2);
    auto* child1 = this->child_towards(*lca, user1.owner_);
    auto* child2 = this->child_towards(*lca, user2.owner_);
    auto lca_end = this->ranges_.at(lca).second;
    auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(lca);

    if (child1 == nullptr && child2 == nullptr) {
        if (loop && user1.index_ < this->loop_reentry(*loop) && user2.index_ >= this->loop_reentry(*loop)) {
            intervals.push_back({user1.index_ + 1, lca_end});
        } else {
            intervals.push_back({user1.index_ + 1, user2.index_});
        }
    } else if (child1 == nullptr) {
        // user1 is a loop header user or an if-else condition read
        if (loop) {
            if (user1.index_ < this->loop_reentry(*loop)) {
                intervals.push_back({user1.index_ + 1, lca_end});
            } else {
                this->entry_intervals(user2, *child2, intervals);
            }
        } else {
            auto* if_else = dynamic_cast<const structured_control_flow::IfElse*>(lca);
            intervals.push_back({user1.index_ + 1, this->ranges_.at(&if_else->at(0).first).first});
            this->entry_intervals(user2, *child2, intervals);
        }
    } else if (child2 == nullptr) {
        // user1 in the loop body, user2 a loop update user
        this->exit_intervals(user1, *child1, intervals);
        intervals.push_back({this->ranges_.at(child1).second, user2.index_});
    } else {
        // Siblings in a sequence
        this->exit_intervals(user1, *child1, intervals);
        intervals.push_back({this->ranges_.at(child1).second, this->ranges_.at(child2).first});
        this->entry_intervals(user2, *child2, intervals);
    }
    return this->collect_uses(intervals, &user1, &user2, true);
}

} // namespace users
} // namespace sdfg
