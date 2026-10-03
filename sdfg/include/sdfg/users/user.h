#pragma once

#include <string>
#include <vector>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace analysis {
class DataDependencyAnalysis;
} // namespace analysis

namespace users {

class Users;
class UsersView;

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
    friend class analysis::DataDependencyAnalysis;

private:
    // Creation order during traversal; users of a control-flow node occupy a contiguous range.
    size_t index_ = 0;
    // Innermost control-flow node whose traversal created this user; null for artificial users.
    structured_control_flow::ControlFlowNode* owner_ = nullptr;

    std::string container_;
    Element* element_;
    Use use_;

    mutable std::vector<data_flow::Subset> subsets_;
    mutable bool subsets_cached_ = false;

public:
    User(const std::string& container, Element* element, Use use);

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
    ForUser(const std::string& container, Element* element, Use use, bool is_init, bool is_condition, bool is_update);

    bool is_init() const;

    bool is_condition() const;

    bool is_update() const;
};

} // namespace users
} // namespace sdfg
