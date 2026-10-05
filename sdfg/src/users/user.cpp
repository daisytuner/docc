#include "sdfg/users/user.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace users {

User::User(const std::string& container, Element* element, Use use)
    : container_(container), use_(use), element_(element) {

      };

Use User::use() const {
    return this->use_;
};

std::string& User::container() {
    return this->container_;
};

Element* User::element() {
    return this->element_;
};

const std::vector<data_flow::Subset>& User::subsets() const {
    if (this->subsets_cached_) {
        return this->subsets_;
    }

    if (this->container_ == "") {
        // No-op user
    } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(this->element_)) {
        auto& graph = access_node->get_parent();
        if (this->use_ == Use::READ || this->use_ == Use::VIEW) {
            for (auto& iedge : graph.out_edges(*access_node)) {
                this->subsets_.push_back(iedge.subset());
            }
        } else if (this->use_ == Use::WRITE || this->use_ == Use::MOVE) {
            for (auto& oedge : graph.in_edges(*access_node)) {
                this->subsets_.push_back(oedge.subset());
            }
        }
    } else if (auto* reduction = dyn_cast<structured_control_flow::Reduce*>(element_)) {
        for (const auto& entry : reduction->reductions()) {
            if (entry.container == container_ && !entry.original_index.is_null()) {
                subsets_.push_back({entry.original_index});
            }
        }
        if (subsets_.empty()) {
            subsets_.push_back({});
        }
    } else {
        // Use of symbol
        this->subsets_.push_back({});
    }

    this->subsets_cached_ = true;
    return this->subsets_;
};

ForUser::ForUser(const std::string& container, Element* element, Use use, bool is_init, bool is_condition, bool is_update)
    : User(container, element, use), is_init_(is_init), is_condition_(is_condition), is_update_(is_update) {

      };

bool ForUser::is_init() const {
    return this->is_init_;
};

bool ForUser::is_condition() const {
    return this->is_condition_;
};

bool ForUser::is_update() const {
    return this->is_update_;
};

} // namespace users
} // namespace sdfg
