#include "sdfg/passes/symbolic/symbol_propagation.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"

namespace sdfg {
namespace passes {

/**
 * @brief Computes the inverse of a simple symbolic expression.
 *
 * This function attempts to compute the inverse of an expression containing
 * a symbol. It handles limited cases:
 * - Addition: lhs + c -> lhs - c
 * - Multiplication: lhs * c -> lhs / c
 *
 * @param lhs The symbol to solve for
 * @param rhs The expression containing the symbol
 * @return The inverse expression, or null if inversion is not possible
 */
symbolic::Expression inverse(const symbolic::Symbol lhs, const symbolic::Expression rhs) {
    if (!symbolic::uses(rhs, lhs)) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::Add>(*rhs)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(rhs);
        auto arg_0 = add->get_args()[0];
        auto arg_1 = add->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::sub(lhs, arg_1);
    } else if (SymEngine::is_a<SymEngine::Mul>(*rhs)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(rhs);
        auto arg_0 = mul->get_args()[0];
        auto arg_1 = mul->get_args()[1];
        if (!symbolic::eq(arg_0, lhs)) {
            std::swap(arg_0, arg_1);
        }
        if (!symbolic::eq(arg_0, lhs)) {
            return SymEngine::null;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg_1)) {
            return SymEngine::null;
        }
        return symbolic::div(lhs, arg_1);
    }

    return SymEngine::null;
};

SymbolPropagation::SymbolPropagation()
    : Pass() {

      };

std::string SymbolPropagation::name() { return "SymbolPropagation"; };

bool SymbolPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    std::unordered_set<data_flow::AccessNode*> replaced_nodes;
    std::unordered_set<std::string> skip;
    for (auto& name : sdfg.containers()) {
        // Criterion: Only transients
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (skip.find(name) != skip.end()) {
            continue;
        }

        // Criterion: Only integers
        auto& type = builder.subject().type(name);
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar || !types::is_integer(scalar->primitive_type())) {
            continue;
        }

        // The symbol will become the LHS (to be replaced)
        auto lhs = symbolic::symbol(name);

        // Collect all reads of the symbol w.r.t to their writes
        auto raw_groups = data_dependency_analysis.defined_by(name);
        for (auto& entry : raw_groups) {
            if (entry.first->use() != analysis::Use::READ) {
                continue;
            }
            auto read = entry.first;

            // Reverse propagation
            if (entry.second.size() == 2) {
                // if (...) { a = 1} else { a = 2 } b = a
                // -> if (...) { a = 1, b = 1} else { a = 2, b = 2 }
                auto write1 = *entry.second.begin();
                auto write2 = *(++entry.second.begin());
                if (write1->container() != write2->container()) {
                    continue;
                }
                auto rhs = symbolic::symbol(write1->container());
                if (data_dependency_analysis.is_undefined_user(*write1) ||
                    data_dependency_analysis.is_undefined_user(*write2)) {
                    continue;
                }
                if (users.num_reads(write1->container()) != 1) {
                    continue;
                }
                if (!dominance_analysis.post_dominates(*read, *write1) ||
                    !dominance_analysis.post_dominates(*read, *write2)) {
                    continue;
                }
                auto transition1 = dyn_cast<structured_control_flow::Transition*>(write1->element());
                auto transition2 = dyn_cast<structured_control_flow::Transition*>(write2->element());
                if (!transition1 || !transition2) {
                    continue;
                }
                auto transition_lhs = dyn_cast<structured_control_flow::Transition*>(read->element());
                if (!transition_lhs) {
                    continue;
                }
                symbolic::Symbol lhs = SymEngine::null;
                for (auto& assign_entry : transition_lhs->assignments()) {
                    if (symbolic::eq(assign_entry.second, rhs)) {
                        lhs = assign_entry.first;
                        break;
                    }
                }
                if (lhs.is_null()) {
                    continue;
                }
                if (transition1->assignments().find(lhs) != transition1->assignments().end() ||
                    transition2->assignments().find(lhs) != transition2->assignments().end()) {
                    continue;
                }

                auto rhs1 = transition1->assignments().at(rhs);
                if (symbolic::uses(rhs1, lhs)) {
                    if (!symbolic::eq(rhs1, lhs)) {
                        continue;
                    }
                }
                auto rhs2 = transition2->assignments().at(rhs);
                if (symbolic::uses(rhs2, lhs)) {
                    if (!symbolic::eq(rhs2, lhs)) {
                        continue;
                    }
                }
                transition1->assignments().insert({lhs, rhs1});
                transition2->assignments().insert({lhs, rhs2});
                transition_lhs->assignments().erase(lhs);
                skip.insert(lhs->get_name());
                skip.insert(rhs->get_name());
                applied = true;
                break;
            }
            // Forward propagation
            else if (entry.second.size() == 1) {
                // Criterion: Write must be a transition
                auto write = *entry.second.begin();
                if (data_dependency_analysis.is_undefined_user(*write)) {
                    continue;
                }
                auto transition = dyn_cast<structured_control_flow::Transition*>(write->element());
                if (!transition) {
                    continue;
                }

                // We now define the rhs (to be propagated expression)
                if (transition->assignments().count(lhs) == 0) {
                    // Reverse propagation already applied
                    continue;
                }
                auto rhs = transition->assignments().at(lhs);

                // Criterion: RHS is not trivial and not recursive
                if (symbolic::eq(lhs, rhs) || symbolic::uses(rhs, lhs)) {
                    continue;
                }

                // Criterion: Write dominates read to not cause data races
                if (!dominance_analysis.dominates(*write, *read)) {
                    continue;
                }

                // Collect all symbols used in the RHS
                std::unordered_set<std::string> rhs_symbols;
                for (auto& sym : symbolic::atoms(rhs)) {
                    if (symbolic::eq(sym, symbolic::__nullptr__())) {
                        continue;
                    }
                    rhs_symbols.insert(sym->get_name());
                }

                auto rhs_modified = rhs;

                // Find dangerous users between write and read
                auto is_dangerous = [&](analysis::User* user) {
                    if (user == write || user == read) {
                        return false;
                    }

                    // Criterion: RHS must dominate modification
                    if (!dominance_analysis.dominates(*write, *user)) {
                        return false;
                    }

                    // Criterion: Modification must dominate read
                    if (dominance_analysis.dominates(*read, *user)) {
                        return false;
                    }

                    return true;
                };
                std::unordered_set<std::string> dangerous_users;
                for (const auto& sym : rhs_symbols) {
                    for (auto* user : users.writes(sym)) {
                        if (is_dangerous(user)) {
                            dangerous_users.insert(sym);
                            break;
                        }
                    }
                    for (auto* user : users.moves(sym)) {
                        if (is_dangerous(user)) {
                            dangerous_users.insert(sym);
                            break;
                        }
                    }
                }
                if (!dangerous_users.empty()) {
                    // RHS' symbols may be written between write and read
                    // We attempt to create the new RHS
                    bool success = true;
                    std::unordered_set<std::string> modified_symbols;
                    auto middle_users = users.all_uses_between(*write, *read);
                    for (auto& user : middle_users) {
                        if (user->use() != analysis::Use::WRITE && user->use() != analysis::Use::MOVE) {
                            continue;
                        }
                        if (rhs_symbols.find(user->container()) == rhs_symbols.end()) {
                            continue;
                        }

                        // Criterion: Symbol is only modified once
                        if (modified_symbols.find(user->container()) != modified_symbols.end()) {
                            success = false;
                            break;
                        }

                        // Criterion: RHS must dominate modification
                        if (!dominance_analysis.dominates(*write, *user)) {
                            success = false;
                            break;
                        }

                        // Criterion: Modification must dominate read
                        if (!dominance_analysis.dominates(*user, *read)) {
                            success = false;
                            break;
                        }

                        // Criterion: Only transitions
                        if (!dyn_cast<structured_control_flow::Transition*>(user->element())) {
                            success = false;
                            break;
                        }
                        auto sym_transition = dyn_cast<structured_control_flow::Transition*>(user->element());
                        auto sym_lhs = symbolic::symbol(user->container());
                        auto sym_rhs = sym_transition->assignments().at(sym_lhs);

                        // Limited to constants
                        for (auto& atom : symbolic::atoms(sym_rhs)) {
                            if (!symbolic::eq(atom, sym_lhs)) {
                                success = false;
                                break;
                            }
                        }
                        if (!success) {
                            break;
                        }

                        auto inv = inverse(sym_lhs, sym_rhs);
                        if (inv == SymEngine::null) {
                            success = false;
                            break;
                        }

                        rhs_modified = symbolic::subs(rhs_modified, sym_lhs, inv);
                        modified_symbols.insert(user->container());
                    }
                    if (!success) {
                        continue;
                    }
                }
                rhs_modified = symbolic::simplify(rhs_modified);

                if (auto transition_stmt = dyn_cast<structured_control_flow::Transition*>(read->element())) {
                    auto& assignments = transition_stmt->assignments();
                    for (auto& entry : assignments) {
                        if (symbolic::uses(entry.second, lhs)) {
                            entry.second = symbolic::subs(entry.second, lhs, rhs_modified);
                            applied = true;
                        }
                    }
                } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(read->element())) {
                    // Criterion: RHS does not use nvptx symbols
                    bool nvptx = false;
                    for (auto& atom : symbolic::atoms(rhs_modified)) {
                        if (symbolic::is_nv(atom)) {
                            nvptx = true;
                            break;
                        }
                    }
                    if (nvptx) {
                        continue;
                    }

                    for (size_t i = 0; i < if_else_stmt->size(); i++) {
                        auto child = if_else_stmt->at(i);
                        if (symbolic::uses(child.second, lhs)) {
                            builder.update_if_else_condition(
                                *if_else_stmt, i, symbolic::subs(child.second, lhs, rhs_modified)
                            );
                            applied = true;
                        }
                    }
                } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(read->element())) {
                    bool used = false;
                    auto subset = memlet->subset();
                    for (auto& dim : subset) {
                        if (symbolic::uses(dim, lhs)) {
                            dim = symbolic::subs(dim, lhs, rhs_modified);
                            used = true;
                        }
                    }
                    if (used) {
                        memlet->set_subset(subset);
                        applied = true;
                    }
                } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(read->element())) {
                    if (SymEngine::is_a<SymEngine::Symbol>(*rhs_modified)) {
                        auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(rhs_modified);
                        if (symbolic::is_nullptr(new_symbol) ||
                            sdfg.type(new_symbol->get_name()).type_id() == types::TypeID::Pointer) {
                            continue;
                        }
                        access_node->data(new_symbol->get_name());
                        applied = true;
                        replaced_nodes.insert(access_node);
                    } else if (SymEngine::is_a<SymEngine::Integer>(*rhs_modified)) {
                        auto new_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs_modified);
                        auto& graph = access_node->get_parent();
                        auto block = static_cast<structured_control_flow::Block*>(graph.get_parent());

                        // Replace with const node
                        auto& const_node =
                            builder
                                .add_constant(*block, std::to_string(new_int->as_int()), type, access_node->debug_info());

                        std::unordered_set<data_flow::Memlet*> replace_edges;
                        for (auto& oedge : graph.out_edges(*access_node)) {
                            builder.add_memlet(
                                *block,
                                const_node,
                                oedge.src_conn(),
                                oedge.dst(),
                                oedge.dst_conn(),
                                oedge.subset(),
                                oedge.base_type(),
                                oedge.debug_info()
                            );
                            replace_edges.insert(&oedge);
                        }
                        for (auto& iedge : graph.in_edges(*access_node)) {
                            builder.add_memlet(
                                *block,
                                iedge.src(),
                                iedge.src_conn(),
                                const_node,
                                iedge.dst_conn(),
                                iedge.subset(),
                                iedge.base_type(),
                                iedge.debug_info()
                            );
                        }

                        for (auto& edge : replace_edges) {
                            builder.remove_memlet(*block, *edge);
                        }
                        builder.remove_node(*block, *access_node);
                        applied = true;
                    }
                } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(read->element())) {
                    for (auto& symbol : library_node->symbols()) {
                        if (symbolic::eq(symbol, lhs)) {
                            library_node->replace(symbol, rhs_modified);
                            applied = true;
                        }
                    }
                } else if (auto for_loop = dyn_cast<structured_control_flow::StructuredLoop*>(read->element())) {
                    auto for_user = dynamic_cast<analysis::ForUser*>(read);
                    if (for_user->is_init() && symbolic::uses(for_loop->init(), lhs)) {
                        auto new_init = symbolic::subs(for_loop->init(), lhs, rhs_modified);
                        new_init = symbolic::simplify(new_init);
                        builder.update_loop(
                            *for_loop, for_loop->indvar(), for_loop->condition(), new_init, for_loop->update()
                        );
                        applied = true;
                    } else if (for_user->is_condition() && symbolic::uses(for_loop->condition(), lhs)) {
                        auto new_condition = symbolic::subs(for_loop->condition(), lhs, rhs_modified);
                        new_condition =
                            SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(symbolic::simplify(new_condition));
                        builder.update_loop(
                            *for_loop, for_loop->indvar(), new_condition, for_loop->init(), for_loop->update()
                        );
                        applied = true;
                    } else if (for_user->is_update() && symbolic::uses(for_loop->update(), lhs)) {
                        auto new_update = symbolic::subs(for_loop->update(), lhs, rhs_modified);
                        new_update = symbolic::simplify(new_update);
                        builder.update_loop(
                            *for_loop, for_loop->indvar(), for_loop->condition(), for_loop->init(), new_update
                        );
                        applied = true;
                    }
                }
            }
        }
    }

    // Post-processing: Merge access nodes and remove dangling nodes
    // Avoid removing elements while iterating above
    for (auto* node : replaced_nodes) {
        builder.merge_siblings(*node);
    }
    for (auto* node : replaced_nodes) {
        auto& graph = node->get_parent();
        auto* block = static_cast<structured_control_flow::Block*>(graph.get_parent());
        for (auto& dnode : graph.data_nodes()) {
            if (graph.in_degree(*dnode) == 0 && graph.out_degree(*dnode) == 0) {
                builder.remove_node(*block, *dnode);
            }
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
