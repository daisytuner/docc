#include "sdfg/passes/structured_control_flow/while_to_for_conversion.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

bool WhileToForConversion::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop
) {
    auto& sdfg = builder.subject();
    auto& body = loop.root();
    if (loop.root().size() < 2) {
        return false;
    }

    // Identify break and continue conditions
    auto end_of_body = body.at(body.size() - 1);
    if (end_of_body.second.size() > 0) {
        return false;
    }
    auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&end_of_body.first);
    if (!if_else_stmt || if_else_stmt->size() != 2) {
        return false;
    }

    bool first_is_continue = false;
    bool first_is_break = false;
    auto& first_branch = if_else_stmt->at(0).first;
    if (first_branch.size() != 1) {
        return false;
    }
    auto first_condition = if_else_stmt->at(0).second;
    if (dyn_cast<structured_control_flow::Break*>(&first_branch.at(0).first)) {
        first_is_break = true;
    } else if (dyn_cast<structured_control_flow::Continue*>(&first_branch.at(0).first)) {
        first_is_continue = true;
    }
    if (!first_is_break && !first_is_continue) {
        return false;
    }

    bool second_is_continue = false;
    bool second_is_break = false;
    auto& second_branch = if_else_stmt->at(1).first;
    if (second_branch.size() != 1) {
        return false;
    }
    auto second_condition = if_else_stmt->at(1).second;
    if (dyn_cast<structured_control_flow::Break*>(&second_branch.at(0).first)) {
        second_is_break = true;
    } else if (dyn_cast<structured_control_flow::Continue*>(&second_branch.at(0).first)) {
        second_is_continue = true;
    }
    if (!second_is_break && !second_is_continue) {
        return false;
    }
    if (first_is_break == second_is_break) {
        return false;
    }

    // Check symbolic expressions
    if (!symbolic::is_true(symbolic::Eq(symbolic::Not(first_condition), second_condition))) {
        return false;
    }

    // Criterion: Exactly one moving iterator, which is written once
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(all_users, body);
    analysis::User* update_write = nullptr;
    for (auto& atom : symbolic::atoms(first_condition)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (symbolic::is_pointer(sym)) {
            return false;
        }
        auto& type = sdfg.type(sym->get_name());
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            return false;
        }

        auto writes = body_users.writes(sym->get_name());
        if (writes.size() > 1) {
            return false;
        } else if (writes.size() == 1) {
            if (update_write != nullptr) {
                return false;
            }
            if (!dyn_cast<structured_control_flow::Transition*>(writes.at(0)->element())) {
                return false;
            }
            update_write = writes.at(0);
        } else {
            continue;
        }
    }
    if (update_write == nullptr) {
        return false;
    }
    auto indvar = symbolic::symbol(update_write->container());
    auto update_element = dyn_cast<structured_control_flow::Transition*>(update_write->element());
    ;
    auto update = update_element->assignments().at(indvar);

    std::unordered_set<std::string> update_symbols;
    for (auto atom : symbolic::atoms(update)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        update_symbols.insert(sym->get_name());
    }

    // Check that we can replace all post-usages of iterator (after increment)
    auto users_after = body_users.all_uses_after(*update_write);
    for (auto use : users_after) {
        if (use->use() == analysis::Use::WRITE && update_symbols.find(use->container()) != update_symbols.end()) {
            return false;
        }
    }

    // No other continue, break or return inside loop body
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&loop.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (dynamic_cast<const structured_control_flow::Break*>(current)) {
            return false;
        } else if (dynamic_cast<const structured_control_flow::Continue*>(current)) {
            return false;
        } else if (dynamic_cast<const structured_control_flow::Return*>(current)) {
            return false;
        }

        if (auto sequence_stmt = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            // Ignore the if_else_stmt
            if (if_else == if_else_stmt) {
                continue;
            }
            for (size_t i = 0; i < if_else->size(); i++) {
                queue.push_back(&if_else->at(i).first);
            }
        }
    }

    return true;
}

void WhileToForConversion::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& parent,
    structured_control_flow::While& loop
) {
    auto& sdfg = builder.subject();
    auto& body = loop.root();

    // Identify break and continue conditions
    auto last_element = body.at(body.size() - 1);
    auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&last_element.first);

    auto first_condition = if_else_stmt->at(0).second;

    bool second_is_break = false;
    auto& second_branch = if_else_stmt->at(1).first;
    auto second_condition = if_else_stmt->at(1).second;
    if (dyn_cast<structured_control_flow::Break*>(&second_branch.at(0).first)) {
        second_is_break = true;
    }
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(all_users, body);
    analysis::User* write_to_indvar = nullptr;
    for (auto& atom : symbolic::atoms(first_condition)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        auto writes = body_users.writes(sym->get_name());
        if (writes.size() == 1) {
            write_to_indvar = writes.at(0);
        } else {
            continue;
        }
    }
    auto update_element = dyn_cast<structured_control_flow::Transition*>(write_to_indvar->element());

    auto indvar = symbolic::symbol(write_to_indvar->container());
    auto update = update_element->assignments().at(indvar);

    // Add temporary symbol for uses after increment
    auto pseudo_indvar = symbolic::symbol(builder.find_new_name(indvar->get_name()));
    builder.add_container(pseudo_indvar->get_name(), sdfg.type(indvar->get_name()));

    // All usages after increment of indvar must be replaced with pseudo_indvar
    auto users_after = body_users.all_uses_after(*write_to_indvar);
    for (auto use : users_after) {
        if (use->container() != indvar->get_name()) {
            continue;
        }

        use->element()->replace(indvar, pseudo_indvar);
    }

    // Remove update from transition
    update_element->assignments().erase(indvar);
    update_element->assignments()[pseudo_indvar] = update;

    if (second_is_break) {
        auto& for_loop = builder.convert_while(parent, loop, indvar, first_condition, indvar, update);
        builder.remove_child(for_loop.root(), for_loop.root().size() - 1);
    } else {
        auto& for_loop = builder.convert_while(parent, loop, indvar, second_condition, indvar, update);
        builder.remove_child(for_loop.root(), for_loop.root().size() - 1);
    }
};

WhileToForConversion::WhileToForConversion()
    : Pass() {

      };

std::string WhileToForConversion::name() { return "WhileToForConversion"; };

bool WhileToForConversion::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Add children to queue
        if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                if (auto match = dyn_cast<structured_control_flow::While*>(&sequence_stmt->at(i).first)) {
                    if (this->can_be_applied(builder, analysis_manager, *match)) {
                        this->apply(builder, analysis_manager, *sequence_stmt, *match);
                        applied = true;
                    }
                }

                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else = dyn_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                queue.push_back(&if_else->at(i).first);
            }
        } else if (auto loop_stmt = dyn_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
