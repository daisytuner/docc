#include "sdfg/passes/structured_control_flow/condition_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace passes {

bool ConditionElimination::eliminate_condition(
    structured_control_flow::Sequence& parent,
    structured_control_flow::IfElse& match_node,
    structured_control_flow::Transition& match_transition
) {
    auto branch = match_node.at(0);
    auto& branch_root = branch.first;
    auto& branch_condition = branch.second;
    auto& loop = static_cast<structured_control_flow::StructuredLoop&>(branch_root.at(0).first);

    auto loop_indvar = loop.indvar();
    auto loop_init = loop.init();
    auto loop_condition = loop.condition();

    // indvar shall not be read afterwards directly since loop init overrides it
    analysis::Users& users = this->analysis_manager_.get<analysis::Users>();
    analysis::UsersView loop_users(users, branch_root);
    if (users.num_reads(loop_indvar->get_name()) > loop_users.reads(loop_indvar->get_name()).size()) {
        return false;
    }

    // If loop condition == true => condition == true && condition == false => loop condition == false
    auto loop_iter0_condition = symbolic::subs(loop_condition, loop_indvar, loop_init);
    if (symbolic::eq(loop_iter0_condition, branch_condition)) {
        // Insert placeholder before if-else
        auto& new_child = this->builder_.add_sequence_before(parent, match_node, match_transition.assignments());

        // Move children of branch to placeholder
        this->builder_.move_children(branch_root, new_child);

        // Remove now empty if-else
        this->builder_.remove_child(parent, parent.index(match_node));

        return true;
    }

    return false;
};

ConditionElimination::
    ConditionElimination(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {

      };


bool ConditionElimination::accept(structured_control_flow::Sequence& node) {
    bool applied = false;
    for (size_t i = 0; i < node.size(); i++) {
        auto& current = node.at(i).first;
        if (auto ifelse = dyn_cast<structured_control_flow::IfElse*>(&current)) {
            if (ifelse->size() != 1) {
                continue;
            }
            auto branch = ifelse->at(0);
            auto& condition = branch.second;

            auto& branch_root = branch.first;
            if (branch_root.size() != 1) {
                continue;
            }
            if (!dyn_cast<structured_control_flow::StructuredLoop*>(&branch_root.at(0).first)) {
                continue;
            }

            if (this->eliminate_condition(node, *ifelse, node.at(i).second)) {
                applied = true;
            }
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
