#include "sdfg/analysis/data_dependency_analysis.h"

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/sets.h"

namespace sdfg {
namespace analysis {

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg, bool detailed)
    : Analysis(sdfg), node_(sdfg.root()), detailed_(detailed) {

      };

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node, bool detailed)
    : Analysis(sdfg), node_(node), detailed_(detailed) {

      };

void DataDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    results_.clear();
    undefined_users_.clear();
    loop_carried_dependencies_.clear();

    std::unordered_set<User*> undefined;
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions;

    visit_sequence(analysis_manager, node_, undefined, open_definitions, closed_definitions);

    for (auto& entry : open_definitions) {
        closed_definitions.insert(entry);
    }

    for (auto& entry : closed_definitions) {
        if (results_.find(entry.first->container()) == results_.end()) {
            results_.insert({entry.first->container(), {}});
        }
        results_.at(entry.first->container()).insert(entry);
    }
};

/****** Visitor API ******/

void DataDependencyAnalysis::visit_block(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Block& block,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    auto& dataflow = block.dataflow();

    for (auto node : dataflow.topological_sort()) {
        if (dynamic_cast<data_flow::ConstantNode*>(node) != nullptr) {
            continue;
        }

        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (!symbolic::is_pointer(symbolic::symbol(access_node->data()))) {
                if (dataflow.in_degree(*node) > 0) {
                    Use use = Use::WRITE;
                    for (auto& iedge : dataflow.in_edges(*access_node)) {
                        if (iedge.type() == data_flow::MemletType::Reference ||
                            iedge.type() == data_flow::MemletType::Dereference_Src) {
                            use = Use::MOVE;
                            break;
                        }
                    }

                    if (use == Use::WRITE) {
                        auto current_user = users.get_user(access_node->data(), access_node, use);

                        // Close open definitions if possible
                        std::unordered_map<User*, std::unordered_set<User*>> to_close;
                        for (auto& user : open_definitions) {
                            if (this->closes(analysis_manager, *user.first, *current_user, false)) {
                                to_close.insert(user);
                            }
                        }
                        for (auto& user : to_close) {
                            open_definitions.erase(user.first);
                            closed_definitions.insert(user);
                        }

                        // Start new open definition
                        open_definitions.insert({current_user, {}});
                    }
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.type() == data_flow::MemletType::Reference ||
                            oedge.type() == data_flow::MemletType::Dereference_Dst) {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    if (use == Use::READ) {
                        auto current_user = users.get_user(access_node->data(), access_node, use);

                        // Assign to open definitions
                        bool found_user = false;
                        bool found_undefined_user = false;
                        for (auto& user : open_definitions) {
                            if (this->depends(analysis_manager, *user.first, *current_user)) {
                                user.second.insert(current_user);
                                found_user = true;
                                found_undefined_user = this->is_undefined_user(*user.first);
                            }
                        }
                        // If no definition found or undefined user found, mark as undefined
                        if (!found_user || found_undefined_user) {
                            undefined.insert(current_user);
                        }
                    }
                }
            }
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(node)) {
            for (auto& symbol : library_node->symbols()) {
                auto current_user = users.get_user(symbol->get_name(), library_node, Use::READ);

                // Assign to open definitions
                bool found_user = false;
                bool found_undefined_user = false;
                for (auto& user : open_definitions) {
                    if (this->depends(analysis_manager, *user.first, *current_user)) {
                        user.second.insert(current_user);
                        found_user = true;
                        found_undefined_user = this->is_undefined_user(*current_user);
                    }
                }
                // If no definition found or undefined user found, mark as undefined
                if (!found_user || found_undefined_user) {
                    undefined.insert(current_user);
                }
            }
        }

        for (auto& oedge : dataflow.out_edges(*node)) {
            std::unordered_set<std::string> used;
            for (auto& dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    used.insert(atom->get_name());
                }
            }
            for (auto& atom : used) {
                auto current_user = users.get_user(atom, &oedge, Use::READ);

                // Assign to open definitions
                bool found_user = false;
                bool found_undefined_user = false;
                for (auto& user : open_definitions) {
                    if (this->depends(analysis_manager, *user.first, *current_user)) {
                        user.second.insert(current_user);
                        found_user = true;
                        found_undefined_user = this->is_undefined_user(*user.first);
                    }
                }
                // If no definition found or undefined user found, mark as undefined
                if (!found_user || found_undefined_user) {
                    undefined.insert(current_user);
                }
            }
        }
    }
}

void DataDependencyAnalysis::visit_for(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& for_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    // Init - Read
    for (auto atom : symbolic::atoms(for_loop.init())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, true);

        // Assign to open definitions
        bool found_user = false;
        bool found_undefined_user = false;
        for (auto& user : open_definitions) {
            if (this->depends(analysis_manager, *user.first, *current_user)) {
                user.second.insert(current_user);
                found_user = true;
                found_undefined_user = this->is_undefined_user(*user.first);
            }
        }
        // If no definition found or undefined user found, mark as undefined
        if (!found_user || found_undefined_user) {
            undefined.insert(current_user);
        }
    }

    // Init - Write
    {
        // Write Induction Variable
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, true);

        // Close open definitions if possible
        std::unordered_map<User*, std::unordered_set<User*>> to_close;
        for (auto& user : open_definitions) {
            if (this->closes(analysis_manager, *user.first, *current_user, true)) {
                to_close.insert(user);
            }
        }
        for (auto& user : to_close) {
            open_definitions.erase(user.first);
            closed_definitions.insert(user);
        }

        // Start new open definition
        open_definitions.insert({current_user, {}});
    }

    // Update - Write
    {
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, false, false, true);
        open_definitions.insert({current_user, {}});
    }

    // Condition - Read
    for (auto atom : symbolic::atoms(for_loop.condition())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, false, true);

        // Assign to open definitions
        bool found_user = false;
        bool found_undefined_user = false;
        for (auto& user : open_definitions) {
            if (this->depends(analysis_manager, *user.first, *current_user)) {
                user.second.insert(current_user);
                found_user = true;
                found_undefined_user = this->is_undefined_user(*user.first);
            }
        }
        // If no definition found or undefined user found, mark as undefined
        if (!found_user || found_undefined_user) {
            undefined.insert(current_user);
        }
    }

    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_for;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_for;
    std::unordered_set<User*> undefined_for;

    // Add assumptions for body
    visit_sequence(analysis_manager, for_loop.root(), undefined_for, open_definitions_for, closed_definitions_for);

    // Update - Read
    for (auto atom : symbolic::atoms(for_loop.update())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, false, false, true);

        // Assign to open definitions
        bool found_user = false;
        bool found_undefined_user = false;
        for (auto& user : open_definitions_for) {
            if (this->depends(analysis_manager, *user.first, *current_user)) {
                user.second.insert(current_user);
                found_user = true;
                found_undefined_user = this->is_undefined_user(*user.first);
            }
        }
        // If no definition found or undefined user found, mark as undefined
        if (!found_user || found_undefined_user) {
            undefined_for.insert(current_user);
        }
    }

    // Merge for with outside
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();

    // Closed definitions are simply merged
    for (auto& entry : closed_definitions_for) {
        closed_definitions.insert(entry);
    }

    // Undefined reads are matched or forwarded
    for (auto open_read : undefined_for) {
        // Simple check: no match or undefined user
        std::unordered_set<User*> frontier;
        bool found = false;
        bool found_undefined_user = false;
        for (auto& entry : open_definitions) {
            if (intersects(*entry.first, *open_read, analysis_manager)) {
                entry.second.insert(open_read);
                found = true;
                found_undefined_user = this->is_undefined_user(*entry.first);
                frontier.insert(entry.first);
            }
        }
        if (!found || found_undefined_user) {
            undefined.insert(open_read);
            continue;
        }

        // Users found, check if they fully cover the read
        bool covered = false;
        for (auto& entry : frontier) {
            if (!dominance_analysis.dominates(*entry, *open_read)) {
                continue;
            }
            bool covers = supersedes_restrictive(*open_read, *entry, analysis_manager);
            if (covers) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            undefined.insert(open_read);
        }
    }

    // Open definitions may close outside open definitions after loop
    std::unordered_set<User*> to_close;
    for (auto& previous : open_definitions) {
        for (auto& user : open_definitions_for) {
            if (this->closes(analysis_manager, *previous.first, *user.first, true)) {
                to_close.insert(previous.first);
                break;
            }
        }
    }
    for (auto& user : to_close) {
        closed_definitions.insert({user, open_definitions.at(user)});
        open_definitions.erase(user);
    }

    // Add loop-carried dependencies
    if (this->detailed_ && for_loop.is_monotonic()) {
        // Case: Can analyze
        bool success = this->loop_carried_dependencies_.insert({&for_loop, {}}).second;
        assert(success);
        auto& dependencies = this->loop_carried_dependencies_.at(&for_loop);

        // Helper to merge deltas into an existing entry for a container
        auto merge_deltas = [](LoopCarriedDependencyInfo& info, const symbolic::maps::DependenceDeltas& new_deltas) {
            if (new_deltas.empty) {
                return;
            }
            if (info.deltas.empty) {
                info.deltas = new_deltas;
                return;
            }
            if (new_deltas.deltas_str.empty() || info.deltas.deltas_str.empty()) {
                // One side has no isl representation — keep non-empty or stay empty-string
                info.deltas.empty = false;
                return;
            }
            // Union the two delta sets
            isl_ctx* ctx = isl_ctx_alloc();
            isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
            isl_set* s1 = isl_set_read_from_str(ctx, info.deltas.deltas_str.c_str());
            isl_set* s2 = isl_set_read_from_str(ctx, new_deltas.deltas_str.c_str());
            if (s1 && s2) {
                isl_set* u = isl_set_union(s1, s2);
                char* str = u ? isl_set_to_str(u) : nullptr;
                if (str) {
                    info.deltas.deltas_str = std::string(str);
                    free(str);
                } else {
                    // Union failed (e.g. mismatched dimensionalities) — degrade gracefully
                    info.deltas.deltas_str = "";
                    info.deltas.dimensions.clear();
                }
                if (u) isl_set_free(u);
            } else {
                if (s1) isl_set_free(s1);
                if (s2) isl_set_free(s2);
                // Parse failed — degrade gracefully
                info.deltas.deltas_str = "";
                info.deltas.dimensions.clear();
            }
            isl_ctx_free(ctx);
        };

        // Case 1: Read-Write between iterations
        for (auto& read : undefined_for) {
            for (auto& write : open_definitions_for) {
                auto deltas = loop_deltas(*write.first, *read, analysis_manager, for_loop);
                if (!deltas.empty) {
                    if (dependencies.find(read->container()) == dependencies.end()) {
                        dependencies[read->container()] =
                            LoopCarriedDependencyInfo{LOOP_CARRIED_DEPENDENCY_READ_WRITE, deltas};
                    } else {
                        merge_deltas(dependencies[read->container()], deltas);
                    }
                    write.second.insert(read);
                }
            }
        }

        // Case 2: Write-Write between iterations
        for (auto& write : open_definitions_for) {
            if (dependencies.find(write.first->container()) != dependencies.end()) {
                continue;
            }
            for (auto& write_2 : open_definitions_for) {
                auto deltas = loop_deltas(*write.first, *write_2.first, analysis_manager, for_loop);
                if (!deltas.empty) {
                    dependencies.insert(
                        {write.first->container(),
                         LoopCarriedDependencyInfo{LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, deltas}}
                    );
                    break;
                }
            }
        }
    } else {
        // Case: Cannot analyze
        bool success = this->loop_carried_dependencies_.insert({&for_loop, {}}).second;
        assert(success);
        auto& dependencies = this->loop_carried_dependencies_.at(&for_loop);

        // Over-Approximation:
        // Add loop-carried dependencies for all open reads to all open writes
        for (auto& read : undefined_for) {
            for (auto& write : open_definitions_for) {
                if (this->depends(analysis_manager, *write.first, *read)) {
                    write.second.insert(read);
                    dependencies.insert(
                        {read->container(),
                         LoopCarriedDependencyInfo{
                             LOOP_CARRIED_DEPENDENCY_READ_WRITE, symbolic::maps::DependenceDeltas{false, "", {}}
                         }}
                    );
                }
            }
        }
        // Add loop-carried dependencies for writes
        for (auto& write : open_definitions_for) {
            if (dependencies.find(write.first->container()) == dependencies.end()) {
                dependencies.insert(
                    {write.first->container(),
                     LoopCarriedDependencyInfo{
                         LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, symbolic::maps::DependenceDeltas{false, "", {}}
                     }}
                );
            }
        }
    }

    // Add open definitions from for to outside
    for (auto& entry : open_definitions_for) {
        open_definitions.insert(entry);
    }
}

void DataDependencyAnalysis::visit_if_else(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::IfElse& if_else,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    // Read Conditions
    for (size_t i = 0; i < if_else.size(); i++) {
        auto child = if_else.at(i).second;
        for (auto atom : symbolic::atoms(child)) {
            auto current_user = users.get_user(atom->get_name(), &if_else, Use::READ);

            bool found_user = false;
            bool found_undefined_user = false;
            for (auto& user : open_definitions) {
                if (this->depends(analysis_manager, *user.first, *current_user)) {
                    user.second.insert(current_user);
                    found_user = true;
                    found_undefined_user = this->is_undefined_user(*user.first);
                }
            }
            // If no definition found or undefined user found, mark as undefined
            if (!found_user || found_undefined_user) {
                undefined.insert(current_user);
            }
        }
    }

    std::vector<std::unordered_set<User*>> undefined_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>> open_definitions_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>> closed_definitionss_branches(if_else.size());
    for (size_t i = 0; i < if_else.size(); i++) {
        auto& child = if_else.at(i).first;
        visit_sequence(
            analysis_manager,
            child,
            undefined_branches.at(i),
            open_definitions_branches.at(i),
            closed_definitionss_branches.at(i)
        );
    }

    // merge partial open reads
    for (size_t i = 0; i < if_else.size(); i++) {
        for (auto& entry : undefined_branches.at(i)) {
            bool found_user = false;
            bool found_undefined_user = false;
            for (auto& user : open_definitions) {
                if (this->depends(analysis_manager, *user.first, *entry)) {
                    user.second.insert(entry);
                    found_user = true;
                    found_undefined_user = this->is_undefined_user(*user.first);
                }
            }
            // If no definition found or undefined user found, mark as undefined
            if (!found_user || found_undefined_user) {
                undefined.insert(entry);
            }
        }
    }

    // merge closed writes
    for (auto& closing : closed_definitionss_branches) {
        for (auto& entry : closing) {
            closed_definitions.insert(entry);
        }
    }

    // Close open reads_after_writes for complete branches
    if (if_else.is_complete()) {
        std::unordered_map<User*, std::unordered_set<User*>> to_close;
        std::unordered_set<std::string> candidates;
        std::unordered_set<std::string> candidates_tmp;

        /* Complete close open reads_after_writes
        1. get candidates from first iteration
        2. iterate over all branches and prune candidates
        3. find prior writes for remaining candidates
        4. close open reads_after_writes for all candidates
        */
        for (auto& entry : open_definitions_branches.at(0)) {
            candidates.insert(entry.first->container());
        }
        for (auto& entry : closed_definitionss_branches.at(0)) {
            candidates.insert(entry.first->container());
        }

        for (size_t i = 1; i < if_else.size(); i++) {
            for (auto& entry : open_definitions_branches.at(i)) {
                if (candidates.find(entry.first->container()) != candidates.end()) {
                    candidates_tmp.insert(entry.first->container());
                }
            }
            candidates.swap(candidates_tmp);
            candidates_tmp.clear();
        }

        for (auto& entry : open_definitions) {
            if (candidates.find(entry.first->container()) != candidates.end()) {
                to_close.insert(entry);
            }
        }

        for (auto& entry : to_close) {
            open_definitions.erase(entry.first);
            closed_definitions.insert(entry);
        }
    } else {
        // Incomplete if-else

        // In order to determine whether a new read is undefined
        // we would need to check whether all open definitions
        // jointly dominate the read.
        // Since this is expensive, we apply a trick:
        // For incomplete if-elses and any newly opened definition in
        // any branch, we add an artificial undefined user for that container.
        // If we encounter this user later, we know that not all branches defined it.
        // Hence, we can mark the read as (partially) undefined.

        for (auto& branch : open_definitions_branches) {
            for (auto& open_definition : branch) {
                auto write = open_definition.first;
                auto artificial_user = std::make_unique<
                    User>(boost::graph_traits<graph::Graph>::null_vertex(), write->container(), nullptr, Use::WRITE);
                this->undefined_users_.push_back(std::move(artificial_user));
                open_definitions.insert({this->undefined_users_.back().get(), {}});
            }
        }
    }

    // Add open definitions from branches to outside
    for (auto& branch : open_definitions_branches) {
        for (auto& entry : branch) {
            open_definitions.insert(entry);
        }
    }
}

void DataDependencyAnalysis::visit_while(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& while_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_while;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_while;
    std::unordered_set<User*> undefined_while;

    visit_sequence(analysis_manager, while_loop.root(), undefined_while, open_definitions_while, closed_definitions_while);

    // Scope-local closed definitions
    for (auto& entry : closed_definitions_while) {
        closed_definitions.insert(entry);
    }

    for (auto open_read : undefined_while) {
        // Over-Approximation: Add loop-carried dependencies for all open reads
        for (auto& entry : open_definitions_while) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
            }
        }

        // Connect to outside
        bool found = false;
        for (auto& entry : open_definitions) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(open_read);
        }
    }

    // Add open definitions from while to outside
    for (auto& entry : open_definitions_while) {
        open_definitions.insert(entry);
    }
}

void DataDependencyAnalysis::visit_return(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Return& return_statement,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    if (return_statement.is_data() && !return_statement.data().empty()) {
        auto current_user = users.get_user(return_statement.data(), &return_statement, Use::READ);

        bool found = false;
        for (auto& user : open_definitions) {
            if (user.first->container() == return_statement.data()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(current_user);
        }
    }

    // close all open reads_after_writes
    for (auto& entry : open_definitions) {
        closed_definitions.insert(entry);
    }
    open_definitions.clear();
}

void DataDependencyAnalysis::visit_sequence(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& sequence,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    for (size_t i = 0; i < sequence.size(); i++) {
        auto child = sequence.at(i);
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child.first)) {
            visit_block(analysis_manager, *block, undefined, open_definitions, closed_definitions);
        } else if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child.first)) {
            visit_for(analysis_manager, *for_loop, undefined, open_definitions, closed_definitions);
        } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
            visit_if_else(analysis_manager, *if_else, undefined, open_definitions, closed_definitions);
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&child.first)) {
            visit_while(analysis_manager, *while_loop, undefined, open_definitions, closed_definitions);
        } else if (auto return_statement = dynamic_cast<structured_control_flow::Return*>(&child.first)) {
            visit_return(analysis_manager, *return_statement, undefined, open_definitions, closed_definitions);
        } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&child.first)) {
            visit_sequence(analysis_manager, *sequence, undefined, open_definitions, closed_definitions);
        }

        // handle transitions read
        for (auto& entry : child.second.assignments()) {
            for (auto& atom : symbolic::atoms(entry.second)) {
                if (symbolic::is_pointer(atom)) {
                    continue;
                }
                auto current_user = users.get_user(atom->get_name(), &child.second, Use::READ);

                bool found = false;
                for (auto& user : open_definitions) {
                    if (user.first->container() == atom->get_name()) {
                        user.second.insert(current_user);
                        found = true;
                    }
                }
                if (!found) {
                    undefined.insert(current_user);
                }
            }
        }

        // handle transitions write
        for (auto& entry : child.second.assignments()) {
            auto current_user = users.get_user(entry.first->get_name(), &child.second, Use::WRITE);

            std::unordered_set<User*> to_close;
            for (auto& user : open_definitions) {
                if (this->closes(analysis_manager, *user.first, *current_user, true)) {
                    to_close.insert(user.first);
                }
            }
            for (auto& user : to_close) {
                closed_definitions.insert({user, open_definitions.at(user)});
                open_definitions.erase(user);
            }
            open_definitions.insert({current_user, {}});
        }
    }
}

symbolic::maps::DependenceDeltas DataDependencyAnalysis::loop_deltas(
    User& previous,
    User& current,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    symbolic::maps::DependenceDeltas empty_result{true, "", {}};

    if (previous.container() != current.container()) {
        return empty_result;
    }
    // Shortcut for scalars — always dependent, but distance is unknown
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return symbolic::maps::DependenceDeltas{false, "", {}};
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return empty_result;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(*current_scope, true);

    // We're using the assumptions from the blocks, where the memory accesses occur
    // However, we need to revert constantness assumptions from the perspective of the loop
    // for which we're checking loop-carried dependencies
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (previous_assumptions.find(loop.indvar()) != previous_assumptions.end()) {
        previous_assumptions.at(loop.indvar()).constant(false);
    }
    if (current_assumptions.find(loop.indvar()) != current_assumptions.end()) {
        current_assumptions.at(loop.indvar()).constant(false);
    }
    for (auto& inner_loop : loop_analysis.descendants(&loop)) {
        if (auto structured_loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(inner_loop)) {
            auto indvar = structured_loop->indvar();
            if (previous_assumptions.find(indvar) != previous_assumptions.end()) {
                previous_assumptions.at(indvar).constant(false);
            }
            if (current_assumptions.find(indvar) != current_assumptions.end()) {
                current_assumptions.at(indvar).constant(false);
            }
        }
    }

    // Collect deltas from all subset pairs and union them
    isl_ctx* union_ctx = nullptr;
    isl_set* accumulated = nullptr;
    std::vector<std::string> result_dimensions;

    for (auto& previous_subset : previous_subsets) {
        for (auto& current_subset : current_subsets) {
            auto deltas = symbolic::maps::dependence_deltas(
                previous_subset, current_subset, loop.indvar(), previous_assumptions, current_assumptions
            );
            if (deltas.empty) {
                continue;
            }
            if (deltas.deltas_str.empty()) {
                // Dependence exists but no isl representation (scalar case)
                if (accumulated) {
                    isl_set_free(accumulated);
                    isl_ctx_free(union_ctx);
                }
                return symbolic::maps::DependenceDeltas{false, "", {}};
            }
            if (!union_ctx) {
                union_ctx = isl_ctx_alloc();
                isl_options_set_on_error(union_ctx, ISL_ON_ERROR_CONTINUE);
                accumulated = isl_set_read_from_str(union_ctx, deltas.deltas_str.c_str());
                result_dimensions = deltas.dimensions;
            } else {
                isl_set* new_set = isl_set_read_from_str(union_ctx, deltas.deltas_str.c_str());
                if (new_set && accumulated) {
                    accumulated = isl_set_union(accumulated, new_set);
                } else if (new_set) {
                    isl_set_free(new_set);
                }
            }
        }
    }

    if (!accumulated) {
        if (union_ctx) {
            isl_ctx_free(union_ctx);
        }
        return empty_result;
    }

    // Serialize the union
    char* str = isl_set_to_str(accumulated);
    if (!str) {
        isl_set_free(accumulated);
        isl_ctx_free(union_ctx);
        return symbolic::maps::DependenceDeltas{false, "", {}};
    }
    std::string union_str(str);
    free(str);

    isl_set_free(accumulated);
    isl_ctx_free(union_ctx);

    return symbolic::maps::DependenceDeltas{false, union_str, result_dimensions};
}

bool DataDependencyAnalysis::
    supersedes_restrictive(User& previous, User& current, analysis::AnalysisManager& analysis_manager) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return false;
    }

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();
    auto previous_scope = Users::scope(&previous);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Check if previous subset is subset of any current subset
    for (auto& previous_subset : previous_subsets) {
        bool found = false;
        for (auto& current_subset : current_subsets) {
            if (symbolic::is_subset(previous_subset, current_subset, previous_assumptions, previous_assumptions)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

bool DataDependencyAnalysis::intersects(User& previous, User& current, analysis::AnalysisManager& analysis_manager) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return true;
    }

    if (!this->detailed_) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto previous_scope = Users::scope(&previous);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Check if any current subset intersects with any previous subset
    bool found = false;
    for (auto& current_subset : current_subsets) {
        for (auto& previous_subset : previous_subsets) {
            if (!symbolic::is_disjoint(current_subset, previous_subset, current_assumptions, previous_assumptions)) {
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }

    return found;
}

bool DataDependencyAnalysis::
    closes(analysis::AnalysisManager& analysis_manager, User& previous, User& current, bool requires_dominance) {
    if (previous.container() != current.container()) {
        return false;
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return false;
    }

    // Check dominance
    if (requires_dominance) {
        auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();
        if (!dominance_analysis.post_dominates(current, previous)) {
            return false;
        }
    }

    // Previous memlets are subsets of current memlets
    auto& type = sdfg_.type(previous.container());
    if (type.type_id() == types::TypeID::Scalar) {
        return true;
    }

    if (!this->detailed_) {
        return false;
    }

    // Collect memlets and assumptions
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto previous_scope = Users::scope(&previous);
    auto current_scope = Users::scope(&current);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    auto& previous_memlets = previous.subsets();
    auto& current_memlets = current.subsets();

    for (auto& subset_ : previous_memlets) {
        bool overwritten = false;
        for (auto& subset : current_memlets) {
            if (symbolic::is_subset(subset_, subset, previous_assumptions, current_assumptions)) {
                overwritten = true;
                break;
            }
        }
        if (!overwritten) {
            return false;
        }
    }

    return true;
}

bool DataDependencyAnalysis::depends(analysis::AnalysisManager& analysis_manager, User& previous, User& current) {
    if (previous.container() != current.container()) {
        return false;
    }

    // Previous memlets are subsets of current memlets
    auto& type = sdfg_.type(previous.container());
    if (type.type_id() == types::TypeID::Scalar) {
        return true;
    }

    if (this->is_undefined_user(previous)) {
        return true;
    }

    if (!this->detailed_) {
        return true;
    }

    // Collect memlets and assumptions
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto previous_scope = Users::scope(&previous);
    auto current_scope = Users::scope(&current);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    auto& previous_memlets = previous.subsets();
    auto& current_memlets = current.subsets();

    bool intersect_any = false;
    for (auto& current_subset : current_memlets) {
        for (auto& previous_subset : previous_memlets) {
            if (!symbolic::is_disjoint(current_subset, previous_subset, current_assumptions, previous_assumptions)) {
                intersect_any = true;
                break;
            }
        }
        if (intersect_any) {
            break;
        }
    }

    return intersect_any;
}

/****** Public API ******/

std::unordered_set<User*> DataDependencyAnalysis::defines(User& write) {
    assert(write.use() == Use::WRITE);
    if (results_.find(write.container()) == results_.end()) {
        return {};
    }
    auto& raws = results_.at(write.container());
    assert(raws.find(&write) != raws.end());

    auto& reads_for_write = raws.at(&write);

    std::unordered_set<User*> reads;
    for (auto& entry : reads_for_write) {
        reads.insert(entry);
    }

    return reads;
};

std::unordered_map<User*, std::unordered_set<User*>> DataDependencyAnalysis::definitions(const std::string& container) {
    if (results_.find(container) == results_.end()) {
        return {};
    }
    return results_.at(container);
};

std::unordered_map<User*, std::unordered_set<User*>> DataDependencyAnalysis::defined_by(const std::string& container) {
    auto reads = this->definitions(container);

    std::unordered_map<User*, std::unordered_set<User*>> read_to_writes_map;
    for (auto& entry : reads) {
        for (auto& read : entry.second) {
            if (read_to_writes_map.find(read) == read_to_writes_map.end()) {
                read_to_writes_map[read] = {};
            }
            read_to_writes_map[read].insert(entry.first);
        }
    }
    return read_to_writes_map;
};

std::unordered_set<User*> DataDependencyAnalysis::defined_by(User& read) {
    assert(read.use() == Use::READ);
    auto definitions = this->definitions(read.container());

    std::unordered_set<User*> writes;
    for (auto& entry : definitions) {
        for (auto& r : entry.second) {
            if (&read == r) {
                writes.insert(entry.first);
            }
        }
    }
    return writes;
};

bool DataDependencyAnalysis::is_undefined_user(User& user) const {
    return user.vertex_ == boost::graph_traits<graph::Graph>::null_vertex();
};

bool DataDependencyAnalysis::available(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_carried_dependencies_.find(&loop) != this->loop_carried_dependencies_.end();
};

const std::unordered_map<std::string, LoopCarriedDependencyInfo>& DataDependencyAnalysis::
    dependencies(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_carried_dependencies_.at(&loop);
};

} // namespace analysis
} // namespace sdfg
