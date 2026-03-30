#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class Users;
class UsersView;
class DominanceAnalysis;
class DataDependencyAnalysis;

enum Use {
    NOP, // No-op
    READ,
    WRITE,
    VIEW,
    MOVE
};

class User {
    friend class Users;
    friend class UsersView;
    friend class DataDependencyAnalysis;
    friend class DominanceAnalysis;

private:
    graph::Vertex vertex_;

    std::string container_;
    Element* element_;
    Use use_;

    mutable std::vector<data_flow::Subset> subsets_;
    mutable bool subsets_cached_ = false;

public:
    User(graph::Vertex vertex, const std::string& container, Element* element, Use use);

    virtual ~User() = default;

    std::string& container();

    Use use() const;

    Element* element();

    const std::vector<data_flow::Subset>& subsets() const;
};

class ForUser : public User {
private:
    bool is_init_;
    bool is_condition_;
    bool is_update_;

public:
    ForUser(
        graph::Vertex vertex,
        const std::string& container,
        Element* element,
        Use use,
        bool is_init,
        bool is_condition,
        bool is_update
    );

    bool is_init() const;

    bool is_condition() const;

    bool is_update() const;
};

class Users : public Analysis {
    friend class AnalysisManager;
    friend class UsersView;
    friend class DominanceAnalysis;

private:
    structured_control_flow::ControlFlowNode& node_;

    graph::Graph graph_;
    User* source_;
    User* sink_;

    std::unordered_map<graph::Vertex, std::unique_ptr<User>, boost::hash<graph::Vertex>> users_;

    // Lookup tables for entries and exits of control flow nodes
    std::unordered_map<const structured_control_flow::ControlFlowNode*, User*> entries_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, User*> exits_;

    struct UserProps {
        std::string container;
        Element* element;
        Use use;
        bool is_init;
        bool is_condition;
        bool is_update;

        bool operator==(const UserProps& other) const {
            return container == other.container && element->element_id() == other.element->element_id() &&
                   use == other.use && is_init == other.is_init && is_condition == other.is_condition &&
                   is_update == other.is_update;
        }
    };

    struct UserPropsHash {
        std::size_t operator()(const UserProps& k) const {
            std::size_t h = 0;
            boost::hash_combine(h, k.container);
            boost::hash_combine(h, k.element->element_id());
            boost::hash_combine(h, static_cast<int>(k.use));
            boost::hash_combine(h, k.is_init);
            boost::hash_combine(h, k.is_condition);
            boost::hash_combine(h, k.is_update);
            return h;
        }
    };

    // Lookup table for users by (container, element, use, is_init, is_condition, is_update)
    std::unordered_map<UserProps, User*, UserPropsHash> users_lookup_;

    // Lookup tables for different use types
    std::unordered_map<std::string, std::list<User*>> reads_;
    std::unordered_map<std::string, std::list<User*>> writes_;
    std::unordered_map<std::string, std::list<User*>> views_;
    std::unordered_map<std::string, std::list<User*>> moves_;

    std::pair<graph::Vertex, graph::Vertex> traverse(data_flow::DataFlowGraph& dataflow);

    std::pair<graph::Vertex, graph::Vertex> traverse(structured_control_flow::ControlFlowNode& node);

    void add_user(std::unique_ptr<User> user);

public:
    Users(StructuredSDFG& sdfg);

    Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

    std::string name() const override { return "Users"; }

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

    std::list<User*> uses() const;

    std::list<User*> uses(const std::string& container) const;

    size_t num_uses(const std::string& container) const;

    std::list<User*> writes() const;

    const std::list<User*>& writes(const std::string& container) const;

    size_t num_writes(const std::string& container) const;

    std::list<User*> reads() const;

    const std::list<User*>& reads(const std::string& container) const;

    size_t num_reads(const std::string& container) const;

    std::list<User*> views() const;

    const std::list<User*>& views(const std::string& container) const;

    size_t num_views(const std::string& container) const;

    std::list<User*> moves() const;

    const std::list<User*>& moves(const std::string& container) const;

    size_t num_moves(const std::string& container) const;

    static structured_control_flow::ControlFlowNode* scope(User* user);

    std::unordered_set<std::string> locals(structured_control_flow::ControlFlowNode& node);

    const std::unordered_set<User*> all_uses_between(User& user1, User& user2);

    const std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();
};

class UsersView {
private:
    Users& users_;
    User* entry_;
    User* exit_;

    std::unordered_set<User*> sub_users_;

public:
    UsersView(Users& users, const structured_control_flow::ControlFlowNode& node);

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

    std::unordered_set<User*> all_uses_between(User& user1, User& user2);

    std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();
};

} // namespace analysis
} // namespace sdfg
