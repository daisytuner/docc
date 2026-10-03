#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/users/legacy_user_visitor.h"
#include "sdfg/users/user.h"

namespace sdfg {
namespace users {

class Users : public analysis::Analysis, protected LegacyUserVisitor {
    friend class analysis::AnalysisManager;
    friend class UsersView;

private:
    structured_control_flow::ControlFlowNode& node_;

    std::vector<std::unique_ptr<User>> users_;

    // Users in traversal order, and each traversed node's [begin, end) slice of it.
    std::vector<User*> order_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::pair<size_t, size_t>> ranges_;
    structured_control_flow::ControlFlowNode* current_owner_ = nullptr;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, bool> may_return_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::vector<size_t>> loop_conditions_;

    // Lookup of users by (element, use, loop part); the few candidates are matched by container name.
    struct UserKey {
        size_t element_id;
        Use use;
        uint8_t loop_part;

        bool operator==(const UserKey& other) const = default;
    };

    struct UserKeyHash {
        std::size_t operator()(const UserKey& k) const {
            std::size_t h = k.element_id;
            boost::hash_combine(h, static_cast<int>(k.use));
            boost::hash_combine(h, k.loop_part);
            return h;
        }
    };

    static UserKey user_key(Element* element, Use use, bool is_init, bool is_condition, bool is_update);

    std::unordered_map<UserKey, std::vector<User*>, UserKeyHash> users_lookup_;

    // Users of each container, by use type (only containers with at least one user).
    struct ContainerUsers {
        std::vector<User*> uses;
        std::vector<User*> reads;
        std::vector<User*> writes;
        std::vector<User*> views;
        std::vector<User*> moves;
    };

    std::unordered_map<std::string, ContainerUsers> container_users_;

    const ContainerUsers& container_users(const std::string& container) const;

    // Enclosing owner and range begin of each node currently being traversed.
    std::vector<std::pair<structured_control_flow::ControlFlowNode*, size_t>> traversal_stack_;

    void on_enter(structured_control_flow::ControlFlowNode& node) override;
    void on_leave(structured_control_flow::ControlFlowNode& node) override;
    void on_use(const std::string& container, Element* element, Use use, LoopPart part) override;

    void add_user(std::unique_ptr<User> user);

    // Structural dominance helpers over regions (control-flow nodes).
    using Node = structured_control_flow::ControlFlowNode;

    bool traversed(const Node& node) const;

    const Node* child_towards(const Node& ancestor, const Node* node) const;

    const Node* common_ancestor(const User& a, const User& b) const;

    bool may_return(const Node& node);

    const std::vector<size_t>& loop_conditions(const structured_control_flow::StructuredLoop& loop);

    bool must_exit(const User& a, const Node& region);

    bool no_escape(const User& b, const Node& region);

    bool post_dominates_entry(const User& a, const Node& region);

    // Structural reachability as [begin, end) slices of order_.
    using Interval = std::pair<size_t, size_t>;

    size_t loop_reentry(const structured_control_flow::StructuredLoop& loop);

    void forward_intervals(const User& user, const Node& stop, bool cut_at_stop, std::vector<Interval>& out);

    void exit_intervals(const User& user, const Node& region, std::vector<Interval>& out);

    void entry_intervals(const User& user, const Node& region, std::vector<Interval>& out);

    std::unordered_set<User*>
    collect_uses(const std::vector<Interval>& intervals, const User* skip1, const User* skip2, bool skip_returns) const;

public:
    Users(StructuredSDFG& sdfg);

    Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

    std::string name() const override {
        return "Users";
    }

    void run(analysis::AnalysisManager& analysis_manager) override;

    bool has_user(
        const std::string& container,
        Element* element,
        Use use,
        bool is_init = false,
        bool is_condition = false,
        bool is_update = false
    );

    User* get_user(
        const std::string& container,
        Element* element,
        Use use,
        bool is_init = false,
        bool is_condition = false,
        bool is_update = false
    );

    /**** Users ****/

    std::vector<User*> uses() const;

    std::vector<User*> uses(const std::string& container) const;

    size_t num_uses(const std::string& container) const;

    std::vector<User*> writes() const;

    const std::vector<User*>& writes(const std::string& container) const;

    size_t num_writes(const std::string& container) const;

    std::vector<User*> reads() const;

    const std::vector<User*>& reads(const std::string& container) const;

    size_t num_reads(const std::string& container) const;

    std::vector<User*> views() const;

    const std::vector<User*>& views(const std::string& container) const;

    size_t num_views(const std::string& container) const;

    std::vector<User*> moves() const;

    const std::vector<User*>& moves(const std::string& container) const;

    size_t num_moves(const std::string& container) const;

    static structured_control_flow::ControlFlowNode* scope(User* user);

    std::unordered_set<std::string> locals(structured_control_flow::ControlFlowNode& node);

    const std::unordered_set<User*> all_uses_between(User& user1, User& user2);

    const std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();

    /// Strict dominance between two users, derived from.
    bool dominates(User& user1, User& user2);

    /// Strict post-dominance between two users, derived from the structured control flow.
    bool post_dominates(User& user1, User& user2);
};

class UsersView {
private:
    Users& users_;
    const structured_control_flow::ControlFlowNode* node_;
    size_t begin_ = 0;
    size_t end_ = 0;

public:
    UsersView(Users& users, const structured_control_flow::ControlFlowNode& node);

    bool contains(const User& user) const;

    /**** Users ****/

    std::vector<User*> uses() const;

    std::vector<User*> uses(const std::string& container) const;

    std::vector<User*> writes() const;

    std::vector<User*> writes(const std::string& container) const;

    std::vector<User*> reads() const;

    std::vector<User*> reads(const std::string& container) const;

    std::vector<User*> views() const;

    std::vector<User*> views(const std::string& container) const;

    std::vector<User*> moves() const;

    std::vector<User*> moves(const std::string& container) const;

    std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();
};

} // namespace users
} // namespace sdfg
