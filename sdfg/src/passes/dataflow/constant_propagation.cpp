#include "sdfg/passes/dataflow/constant_propagation.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"

namespace sdfg {
namespace passes {

ConstantPropagation::ConstantPropagation() : Pass() {};

std::string ConstantPropagation::name() { return "ConstantPropagation"; };

std::unordered_set<analysis::User*>
inputs(const std::string& container, data_flow::AccessNode* access_node, analysis::Users& users) {
    std::unordered_set<analysis::User*> inputs;

    auto& graph = access_node->get_parent();
    if (graph.in_degree(*access_node) != 1) {
        return {};
    }
    auto& iedge = *graph.in_edges(*access_node).begin();

    auto& src = iedge.src();
    if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&src)) {
        for (auto& iedge : graph.in_edges(*tasklet)) {
            auto& src_node = static_cast<data_flow::AccessNode&>(iedge.src());
            if (dynamic_cast<data_flow::ConstantNode*>(&src_node) != nullptr) {
                continue;
            }

            inputs.insert(users.get_user(src_node.data(), &src_node, analysis::Use::READ));
        }
    } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&src)) {
        if (iedge.type() == data_flow::MemletType::Dereference_Src) {
            inputs.insert(users.get_user(access_node->data(), access_node, analysis::Use::READ));
        } else if (iedge.type() == data_flow::MemletType::Reference) {
            inputs.insert(users.get_user(access_node->data(), access_node, analysis::Use::VIEW));
        }
    }

    return inputs;
}

std::unordered_set<analysis::User*>
inputs(const std::string& container, structured_control_flow::Transition* transition, analysis::Users& users) {
    std::unordered_set<analysis::User*> inputs;
    auto& assign = transition->assignments().at(symbolic::symbol(container));
    for (auto& sym : symbolic::atoms(assign)) {
        if (symbolic::eq(sym, symbolic::__nullptr__())) {
            continue;
        }
        inputs.insert(users.get_user(sym->get_name(), transition, analysis::Use::READ));
    }
    return inputs;
}

std::unordered_set<analysis::User*> inputs(analysis::User& user, analysis::Users& users) {
    if (auto access_node = dynamic_cast<data_flow::AccessNode*>(user.element())) {
        return inputs(user.container(), access_node, users);
    } else if (auto transition = dyn_cast<structured_control_flow::Transition*>(user.element())) {
        return inputs(user.container(), transition, users);
    } else {
        return {};
    }
}

bool ConstantPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

    // We seek to find two identical definitions of the same container
    for (auto& name : sdfg.containers()) {
        std::unordered_set<analysis::User*> defines;
        if (users.writes(name).size() == 2 && users.moves(name).size() == 0) {
            for (auto& write : users.writes(name)) {
                defines.insert(write);
            }
        } else if (users.moves(name).size() == 2 && users.writes(name).size() == 0) {
            for (auto& move : users.moves(name)) {
                defines.insert(move);
            }
        } else {
            continue;
        }
        auto define1 = *defines.begin();
        auto define2 = *(++defines.begin());

        // Criterion: One dominates the other
        if (!dominance_analysis.dominates(*define1, *define2)) {
            std::swap(define1, define2);
        }
        if (!dominance_analysis.dominates(*define1, *define2)) {
            continue;
        }

        // Criterion: Prove identical subsets of definitions
        auto subsets1 = define1->subsets();
        auto subsets2 = define2->subsets();
        if (subsets1.size() != 1 || subsets2.size() != 1) {
            continue;
        }
        if (subsets1.begin()->size() != subsets2.begin()->size()) {
            continue;
        }
        bool identical_subsets = true;
        for (size_t i = 0; i < subsets1.begin()->size(); i++) {
            auto dim1 = subsets1.begin()->at(i);
            auto dim2 = subsets2.begin()->at(i);
            if (!symbolic::eq(dim1, dim2)) {
                identical_subsets = false;
                break;
            }
            std::unordered_set<std::string> symbols;
            for (auto& sym : symbolic::atoms(dim1)) {
                symbols.insert(sym->get_name());
            }
            for (auto& user : users.all_uses_between(*define1, *define2)) {
                if (user->use() == analysis::Use::READ) {
                    continue;
                }
                if (symbols.find(user->container()) != symbols.end()) {
                    identical_subsets = false;
                    break;
                }
            }
        }
        if (!identical_subsets) {
            continue;
        }

        // Criterion: Provide identical rhs/inputs of definitions
        auto inputs1 = inputs(*define1, users);
        if (inputs1.empty()) {
            continue;
        }
        auto inputs2 = inputs(*define2, users);
        if (inputs2.empty()) {
            continue;
        }
        if (inputs1.size() != inputs2.size()) {
            continue;
        }

        bool identical_inputs = true;
        for (auto& input : inputs1) {
            // Recursion
            if (input->container() == name) {
                identical_inputs = false;
                break;
            }

            // If input is written, it must happen before define1
            for (auto& write_user : users.writes(input->container())) {
                if (!dominance_analysis.dominates(*write_user, *define1)) {
                    identical_inputs = false;
                    break;
                }
            }

            // If input is moved, it must happen before define1
            for (auto& move_user : users.moves(input->container())) {
                if (!dominance_analysis.dominates(*move_user, *define1)) {
                    identical_inputs = false;
                    break;
                }
            }

            // If input has views, it must be the definitions
            for (auto& view : users.views(input->container())) {
                auto& viewed_node = dynamic_cast<data_flow::AccessNode&>(*view->element());
                auto& graph = viewed_node.get_parent();
                if (graph.out_degree(viewed_node) != 1) {
                    continue;
                }
                auto& viewing_node = (*graph.out_edges(viewed_node).begin()).dst();
                if (&viewing_node != define1->element() && &viewing_node != define2->element()) {
                    identical_inputs = false;
                }
            }
            if (!identical_inputs) {
                break;
            }

            // Find identical input in inputs2
            analysis::User* input2 = nullptr;
            for (auto& inp : inputs2) {
                if (input->container() == inp->container()) {
                    input2 = inp;
                    break;
                }
            }
            if (input2 == nullptr) {
                identical_inputs = false;
                break;
            }

            // same subsets
            if (input->subsets().size() != 1 || input2->subsets().size() != 1) {
                identical_inputs = false;
                break;
            }

            auto subset1 = *input->subsets().begin();
            auto subset2 = *input2->subsets().begin();
            if (subset1.size() != subset2.size()) {
                identical_inputs = false;
                break;
            }
            for (size_t i = 0; i < subset1.size(); i++) {
                auto dim1 = subset1[i];
                auto dim2 = subset2[i];
                if (!symbolic::eq(dim1, dim2)) {
                    identical_inputs = false;
                    break;
                }
                std::unordered_set<std::string> symbols;
                for (auto& sym : symbolic::atoms(dim1)) {
                    symbols.insert(sym->get_name());
                }
                for (auto& user : users.all_uses_between(*define1, *define2)) {
                    if (user->use() == analysis::Use::READ) {
                        continue;
                    }
                    if (symbols.find(user->container()) != symbols.end()) {
                        identical_inputs = false;
                        break;
                    }
                }
            }
        }
        if (!identical_inputs) {
            continue;
        }

        // Eliminate the dominated definition
        auto write = define2->element();
        if (auto transition = dyn_cast<structured_control_flow::Transition*>(write)) {
            transition->assignments().erase(symbolic::symbol(name));
            applied = true;
        } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(write)) {
            auto& graph = access_node->get_parent();
            auto& block = dyn_cast<structured_control_flow::Block&>(*graph.get_parent());
            builder.clear_node(block, *access_node);
            applied = true;
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
