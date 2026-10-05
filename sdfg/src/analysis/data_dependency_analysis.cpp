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
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/sets.h"

namespace sdfg {
namespace analysis {

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg), node_(sdfg.root()) {};

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node)
    : Analysis(sdfg), node_(node) {};

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::StructuredLoop& loop)
    : Analysis(sdfg), node_(loop.root()), loop_(&loop) {};

AssumptionsAnalysis& DataDependencyAnalysis::ensure_detailed_assumptions(analysis::AnalysisManager& analysis_manager) {
    if (!detailed_assumptions_) {
        detailed_assumptions_ = std::make_unique<AssumptionsAnalysis>(sdfg_, /*with_branch_conditions=*/true);
        detailed_assumptions_->run(analysis_manager);
    }
    return *detailed_assumptions_;
}

void DataDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    results_.clear();
    undefined_users_.clear();
    loop_boundaries_.clear();
    scalar_containers_.clear();

    // Reset the detailed assumptions-analysis cache. It will be lazily
    // (re)constructed by `ensure_detailed_assumptions()` on first symbolic
    // subset/disjointness query - which only happens when `detailed_` is
    // set. The cheap, branch-condition-less manager-cached instance is
    // insufficient for those queries because halo-style IfElse guards
    // (e.g. `2*wout + kw < 226`) only register their coupled constraints
    // in the detailed form.
    detailed_assumptions_.reset();

    std::unordered_set<User*> undefined;
    OpenDefinitions open_definitions;
    Definitions closed_definitions;

    if (loop_ != nullptr) {
        visit_for_impl(analysis_manager, *loop_, undefined, open_definitions, closed_definitions);
    } else {
        visit_sequence_impl(analysis_manager, node_, undefined, open_definitions, closed_definitions);
    }

    for (auto& [container, group] : open_definitions) {
        for (auto& entry : group) {
            closed_definitions.insert(std::move(entry));
        }
    }

    for (auto& entry : closed_definitions) {
        results_[entry.first->container()].insert(std::move(entry));
    }
};

/****** Visitor API ******/

DataDependencyAnalysis::OpenDefinitions DataDependencyAnalysis::group_by_container(const Definitions& definitions) {
    OpenDefinitions grouped;
    for (auto& entry : definitions) {
        grouped[entry.first->container()].insert(entry);
    }
    return grouped;
}

DataDependencyAnalysis::Definitions DataDependencyAnalysis::flatten(const OpenDefinitions& definitions) {
    Definitions flat;
    for (auto& [container, group] : definitions) {
        flat.insert(group.begin(), group.end());
    }
    return flat;
}

namespace {

template<typename OpenDefinitions>
auto* find_group(OpenDefinitions& open_definitions, const std::string& container) {
    using Group = typename OpenDefinitions::mapped_type;
    auto it = open_definitions.find(container);
    return it == open_definitions.end() ? static_cast<Group*>(nullptr) : &it->second;
}

} // namespace

#define DDA_PUBLIC_VISIT(name, node_type)                                            \
    void DataDependencyAnalysis::name(                                               \
        analysis::AnalysisManager& analysis_manager,                                 \
        node_type& node,                                                             \
        std::unordered_set<User*>& undefined,                                        \
        Definitions& open_definitions,                                               \
        Definitions& closed_definitions                                              \
    ) {                                                                              \
        auto grouped = group_by_container(open_definitions);                         \
        name##_impl(analysis_manager, node, undefined, grouped, closed_definitions); \
        open_definitions = flatten(grouped);                                         \
    }

DDA_PUBLIC_VISIT(visit_block, structured_control_flow::Block)
DDA_PUBLIC_VISIT(visit_assignment_block, structured_control_flow::AssignmentBlock)
DDA_PUBLIC_VISIT(visit_for, structured_control_flow::StructuredLoop)
DDA_PUBLIC_VISIT(visit_if_else, structured_control_flow::IfElse)
DDA_PUBLIC_VISIT(visit_while, structured_control_flow::While)
DDA_PUBLIC_VISIT(visit_return, structured_control_flow::Return)
DDA_PUBLIC_VISIT(visit_sequence, structured_control_flow::Sequence)

#undef DDA_PUBLIC_VISIT

void DataDependencyAnalysis::visit_block_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Block& block,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    auto& dataflow = block.dataflow();

    // Assign a read to the matching open definitions; returns {found, found_undefined}.
    auto assign_read = [&](User& current_user, bool undefined_from_current) {
        bool found_user = false;
        bool found_undefined_user = false;
        if (auto* group = find_group(open_definitions, current_user.container())) {
            for (auto& user : *group) {
                if (this->depends(analysis_manager, *user.first, current_user)) {
                    user.second.insert(&current_user);
                    found_user = true;
                    found_undefined_user = this->is_undefined_user(undefined_from_current ? current_user : *user.first);
                }
            }
        }
        return std::make_pair(found_user, found_undefined_user);
    };

    for (const auto* const_node : dataflow.topological_order()) {
        auto* node = const_cast<data_flow::DataFlowNode*>(const_node);
        if (dynamic_cast<data_flow::ConstantNode*>(node) != nullptr) {
            continue;
        }

        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            auto& access_node_type = sdfg_.type(access_node->data());
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
                        auto& group = open_definitions[current_user->container()];
                        Definitions to_close;
                        for (auto& user : group) {
                            if (this->closes(analysis_manager, *user.first, *current_user, false)) {
                                to_close.insert(user);
                            }
                        }
                        for (auto& user : to_close) {
                            group.erase(user.first);
                            closed_definitions.insert(user);
                        }

                        // Start new open definition
                        group.insert({current_user, {}});
                    }
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (access_node_type.type_id() == types::TypeID::Pointer &&
                            oedge.is_src_pointed_to_address_leak(access_node_type)) {
                            if (auto* libConsumer = dynamic_cast<data_flow::LibraryNode*>(&oedge.dst())) {
                                auto access = libConsumer->pointer_access_type(oedge);
                                if (!access || !access->no_capture()) {
                                    use = Use::MOVE; // we know nothing and cannot say anything reliable. Consider the
                                                     // libNode having side effects
                                    break;
                                } else {
                                    if (access->may_contain_reads()) {
                                        // TODO let handle as a normal read
                                        use = Use::VIEW;
                                    } else if (access->may_contain_writes()) {
                                        // handle as a normal write
                                        use = Use::WRITE;
                                        break;
                                    }
                                }
                            } else {
                                use = Use::VIEW;
                                break;
                            }
                        } else if (oedge.type() == data_flow::MemletType::Reference ||
                                   oedge.type() == data_flow::MemletType::Dereference_Dst) {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    // A container feeding a library node's write-only pointer input is written
                    // *through* that pointer. analysis::Users applies this regardless of the access
                    // node's element type (e.g. an Array shared buffer staged by a TileCopyNode's
                    // _dst); mirror it here so the get_user(READ) lookup below never diverges from
                    // what Users registered (a divergence throws std::out_of_range).
                    if (use == Use::READ) {
                        for (auto& oedge : dataflow.out_edges(*access_node)) {
                            auto* lib = dynamic_cast<data_flow::LibraryNode*>(&oedge.dst());
                            if (lib == nullptr) {
                                continue;
                            }
                            auto meta = lib->pointer_access_type(oedge);
                            if (meta && meta->may_contain_writes() && !meta->may_contain_reads()) {
                                use = Use::WRITE;
                                break;
                            }
                        }
                    }

                    if (use == Use::READ) {
                        auto current_user = users.get_user(access_node->data(), access_node, use);

                        auto [found_user, found_undefined_user] = assign_read(*current_user, false);
                        // If no definition found, undefined user found, or
                        // the read's subset footprint is only partially
                        // covered by the matching open writers, mark as
                        // upward-exposed (undefined). `depends` uses
                        // intersect-any across the cartesian product of
                        // subsets, which silently swallows partial-cover
                        // multi-subset reads (e.g. an access node with two
                        // out-edges where only one edge is killed inside
                        // the current scope).
                        if (!found_user || found_undefined_user ||
                            !this->fully_covered(
                                analysis_manager, *current_user, find_group(open_definitions, current_user->container())
                            )) {
                            undefined.insert(current_user);
                        }
                    }
                }
            }
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(node)) {
            for (auto& symbol : library_node->symbols()) {
                auto current_user = users.get_user(symbol->get_name(), library_node, Use::READ);

                auto [found_user, found_undefined_user] = assign_read(*current_user, true);
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

                auto [found_user, found_undefined_user] = assign_read(*current_user, false);
                // If no definition found or undefined user found, mark as undefined
                if (!found_user || found_undefined_user) {
                    undefined.insert(current_user);
                }
            }
        }
    }
}

void DataDependencyAnalysis::visit_assignment_block_impl(
    AnalysisManager& analysis_manager,
    structured_control_flow::AssignmentBlock& assignments,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    // handle transitions read
    for (auto& entry : assignments.assignments()) {
        for (auto& atom : symbolic::atoms(entry.second)) {
            if (symbolic::is_pointer(atom)) {
                continue;
            }
            auto current_user = users.get_user(atom->get_name(), &assignments, Use::READ);

            bool found = false;
            if (auto* group = find_group(open_definitions, atom->get_name())) {
                for (auto& user : *group) {
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
    for (auto& entry : assignments.assignments()) {
        auto current_user = users.get_user(entry.first->get_name(), &assignments, Use::WRITE);

        auto& group = open_definitions[current_user->container()];
        std::unordered_set<User*> to_close;
        for (auto& user : group) {
            if (this->closes(analysis_manager, *user.first, *current_user, true)) {
                to_close.insert(user.first);
            }
        }
        for (auto& user : to_close) {
            closed_definitions.insert({user, group.at(user)});
            group.erase(user);
        }
        group.insert({current_user, {}});
    }
}

void DataDependencyAnalysis::visit_for_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& for_loop,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    // Assign a read to the matching definitions of `open`; marks it undefined in `undefined_target` if unmatched.
    auto assign_read = [&](User* current_user, OpenDefinitions& open, std::unordered_set<User*>& undefined_target) {
        bool found_user = false;
        bool found_undefined_user = false;
        if (auto* group = find_group(open, current_user->container())) {
            for (auto& user : *group) {
                if (this->depends(analysis_manager, *user.first, *current_user)) {
                    user.second.insert(current_user);
                    found_user = true;
                    found_undefined_user = this->is_undefined_user(*user.first);
                }
            }
        }
        // If no definition found or undefined user found, mark as undefined
        if (!found_user || found_undefined_user) {
            undefined_target.insert(current_user);
        }
    };

    // Init - Read
    for (auto atom : symbolic::atoms(for_loop.init())) {
        assign_read(users.get_user(atom->get_name(), &for_loop, Use::READ, true), open_definitions, undefined);
    }

    // Init - Write
    {
        // Write Induction Variable
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, true);

        // Close open definitions if possible
        auto& group = open_definitions[current_user->container()];
        Definitions to_close;
        for (auto& user : group) {
            if (this->closes(analysis_manager, *user.first, *current_user, true)) {
                to_close.insert(user);
            }
        }
        for (auto& user : to_close) {
            group.erase(user.first);
            closed_definitions.insert(user);
        }

        // Start new open definition
        group.insert({current_user, {}});
    }

    // Update - Write
    {
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, false, false, true);
        open_definitions[current_user->container()].insert({current_user, {}});
    }

    // Condition - Read
    for (auto atom : symbolic::atoms(for_loop.condition())) {
        assign_read(users.get_user(atom->get_name(), &for_loop, Use::READ, false, true), open_definitions, undefined);
    }

    OpenDefinitions open_definitions_for;
    Definitions closed_definitions_for;
    std::unordered_set<User*> undefined_for;

    // Packed bodies use partials, but the final combine still reads and writes the original accumulator.
    if (auto* reduction = dyn_cast<structured_control_flow::Reduce*>(&for_loop)) {
        for (const auto& entry : reduction->reductions()) {
            if (entry.original_index.is_null()) {
                continue;
            }
            undefined_for.insert(users.get_user(entry.container, reduction, Use::READ));
            open_definitions_for[entry.container]
                .emplace(users.get_user(entry.container, reduction, Use::WRITE), std::unordered_set<User*>{});
            for (const auto& symbol : symbolic::atoms(entry.original_index)) {
                undefined_for.insert(users.get_user(symbol->get_name(), reduction, Use::READ));
            }
        }
    }

    // Add assumptions for body
    visit_sequence_impl(analysis_manager, for_loop.root(), undefined_for, open_definitions_for, closed_definitions_for);

    // Update - Read
    for (auto atom : symbolic::atoms(for_loop.update())) {
        assign_read(
            users.get_user(atom->get_name(), &for_loop, Use::READ, false, false, true),
            open_definitions_for,
            undefined_for
        );
    }

    // Merge for with outside

    // Closed definitions are simply merged
    for (auto& entry : closed_definitions_for) {
        closed_definitions.insert(std::move(entry));
    }

    // Undefined reads are matched or forwarded
    for (auto open_read : undefined_for) {
        // Simple check: no match or undefined user
        std::unordered_set<User*> frontier;
        bool found = false;
        bool found_undefined_user = false;
        if (auto* group = find_group(open_definitions, open_read->container())) {
            for (auto& entry : *group) {
                if (intersects(*entry.first, *open_read, analysis_manager)) {
                    entry.second.insert(open_read);
                    found = true;
                    found_undefined_user = this->is_undefined_user(*entry.first);
                    frontier.insert(entry.first);
                }
            }
        }
        if (!found || found_undefined_user) {
            undefined.insert(open_read);
            continue;
        }

        // Users found, check if they fully cover the read
        bool covered = false;
        for (auto& entry : frontier) {
            if (!users.dominates(*entry, *open_read)) {
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
    for (auto& [container, group] : open_definitions) {
        auto* group_for = find_group(open_definitions_for, container);
        if (group_for == nullptr) {
            continue;
        }
        std::unordered_set<User*> to_close;
        for (auto& previous : group) {
            for (auto& user : *group_for) {
                if (this->closes(analysis_manager, *previous.first, *user.first, true)) {
                    to_close.insert(previous.first);
                    break;
                }
            }
        }
        for (auto& user : to_close) {
            closed_definitions.insert({user, group.at(user)});
            group.erase(user);
        }
    }

    // Cross-iteration linkage for SCALARS only.
    //
    // An in-body upward-exposed read may be satisfied (in some non-first
    // iteration) by an in-body open writer. We must record those
    // (write -> read) edges in `open_definitions_for` so that public
    // queries like `defined_by(read)` return the in-body writer in
    // addition to any pre-loop write -- otherwise consumers (e.g.
    // SymbolPropagation) treat the read as having a single dominating
    // definition and incorrectly fold the loop-carried value away.
    //
    // We restrict to scalar containers because:
    //   - For scalars, every access touches the same cell, so the
    //     prior-iteration writer always reaches the in-body read --
    //     this is sound and precise.
    //   - For arrays, deciding whether a writer flows across iterations
    //     requires precise per-pair delta analysis (done in LCDA).
    //     Adding edges via the over-approximate `depends` predicate
    //     would create spurious in-body RAW links that pollute
    //     consumers reasoning about precise dataflow on arrays.
    //
    // This must run BEFORE the snapshot below and BEFORE merging into the
    // outer `open_definitions`, since both perform by-value copies of the
    // (writer -> readers) sets we are mutating.
    for (auto* open_read : undefined_for) {
        if (!this->is_scalar(open_read->container())) {
            continue;
        }
        auto* group_for = find_group(open_definitions_for, open_read->container());
        if (group_for == nullptr) {
            continue;
        }
        for (auto& write_entry : *group_for) {
            if (this->is_undefined_user(*write_entry.first)) {
                continue;
            }
            write_entry.second.insert(open_read);
        }
    }

    // Snapshot loop boundary sets so LoopCarriedDependencyAnalysis can compute LCDs.
    loop_boundaries_[&for_loop] = std::make_pair(std::move(undefined_for), flatten(open_definitions_for));

    // Add open definitions from for to outside
    for (auto& [container, group_for] : open_definitions_for) {
        auto& group = open_definitions[container];
        for (auto& entry : group_for) {
            group.insert(std::move(entry));
        }
    }
}

void DataDependencyAnalysis::visit_if_else_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::IfElse& if_else,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    auto assign_read = [&](User* current_user) {
        bool found_user = false;
        bool found_undefined_user = false;
        if (auto* group = find_group(open_definitions, current_user->container())) {
            for (auto& user : *group) {
                if (this->depends(analysis_manager, *user.first, *current_user)) {
                    user.second.insert(current_user);
                    found_user = true;
                    found_undefined_user = this->is_undefined_user(*user.first);
                }
            }
        }
        // If no definition found or undefined user found, mark as undefined
        if (!found_user || found_undefined_user) {
            undefined.insert(current_user);
        }
    };

    // Read Conditions
    for (size_t i = 0; i < if_else.size(); i++) {
        auto child = if_else.at(i).second;
        for (auto atom : symbolic::atoms(child)) {
            assign_read(users.get_user(atom->get_name(), &if_else, Use::READ));
        }
    }

    std::vector<std::unordered_set<User*>> undefined_branches(if_else.size());
    std::vector<OpenDefinitions> open_definitions_branches(if_else.size());
    std::vector<Definitions> closed_definitionss_branches(if_else.size());
    for (size_t i = 0; i < if_else.size(); i++) {
        auto& child = if_else.at(i).first;
        visit_sequence_impl(
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
            assign_read(entry);
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
        std::unordered_set<std::string> candidates;
        std::unordered_set<std::string> candidates_tmp;

        /* Complete close open reads_after_writes
        1. get candidates from first iteration
        2. iterate over all branches and prune candidates
        3. find prior writes for remaining candidates
        4. close open reads_after_writes for all candidates
        */
        for (auto& [container, group] : open_definitions_branches.at(0)) {
            if (!group.empty()) {
                candidates.insert(container);
            }
        }
        for (auto& entry : closed_definitionss_branches.at(0)) {
            candidates.insert(entry.first->container());
        }

        for (size_t i = 1; i < if_else.size(); i++) {
            for (auto& [container, group] : open_definitions_branches.at(i)) {
                if (!group.empty() && candidates.find(container) != candidates.end()) {
                    candidates_tmp.insert(container);
                }
            }
            candidates.swap(candidates_tmp);
            candidates_tmp.clear();
        }

        for (auto& container : candidates) {
            if (auto* group = find_group(open_definitions, container)) {
                for (auto& entry : *group) {
                    closed_definitions.insert(entry);
                }
                group->clear();
            }
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
            for (auto& [container, group] : branch) {
                for (size_t k = 0; k < group.size(); k++) {
                    auto artificial_user = std::make_unique<User>(container, nullptr, Use::WRITE);
                    this->undefined_users_.push_back(std::move(artificial_user));
                    open_definitions[container].insert({this->undefined_users_.back().get(), {}});
                }
            }
        }
    }

    // Add open definitions from branches to outside
    for (auto& branch : open_definitions_branches) {
        for (auto& [container, group_branch] : branch) {
            auto& group = open_definitions[container];
            for (auto& entry : group_branch) {
                group.insert(entry);
            }
        }
    }
}

void DataDependencyAnalysis::visit_while_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& while_loop,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    OpenDefinitions open_definitions_while;
    Definitions closed_definitions_while;
    std::unordered_set<User*> undefined_while;

    visit_sequence_impl(
        analysis_manager, while_loop.root(), undefined_while, open_definitions_while, closed_definitions_while
    );

    // Scope-local closed definitions
    for (auto& entry : closed_definitions_while) {
        closed_definitions.insert(entry);
    }

    for (auto open_read : undefined_while) {
        // Over-Approximation: Add loop-carried dependencies for all open reads
        if (auto* group = find_group(open_definitions_while, open_read->container())) {
            for (auto& entry : *group) {
                entry.second.insert(open_read);
            }
        }

        // Connect to outside
        bool found = false;
        if (auto* group = find_group(open_definitions, open_read->container())) {
            for (auto& entry : *group) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(open_read);
        }
    }

    // Add open definitions from while to outside
    for (auto& [container, group_while] : open_definitions_while) {
        auto& group = open_definitions[container];
        for (auto& entry : group_while) {
            group.insert(entry);
        }
    }
}

void DataDependencyAnalysis::visit_return_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Return& return_statement,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    auto& users = analysis_manager.get<analysis::Users>();

    if (return_statement.is_data() && !return_statement.data().empty()) {
        auto current_user = users.get_user(return_statement.data(), &return_statement, Use::READ);

        bool found = false;
        if (auto* group = find_group(open_definitions, return_statement.data())) {
            for (auto& user : *group) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(current_user);
        }
    }

    // close all open reads_after_writes
    for (auto& [container, group] : open_definitions) {
        for (auto& entry : group) {
            closed_definitions.insert(entry);
        }
    }
    open_definitions.clear();
}

void DataDependencyAnalysis::visit_sequence_impl(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& sequence,
    std::unordered_set<User*>& undefined,
    OpenDefinitions& open_definitions,
    Definitions& closed_definitions
) {
    for (size_t i = 0; i < sequence.size(); i++) {
        auto& child = sequence.at(i);
        if (auto block = dyn_cast<structured_control_flow::Block*>(&child)) {
            visit_block_impl(analysis_manager, *block, undefined, open_definitions, closed_definitions);
        } else if (auto assignments = dyn_cast<structured_control_flow::AssignmentBlock*>(&child)) {
            visit_assignment_block_impl(analysis_manager, *assignments, undefined, open_definitions, closed_definitions);
        } else if (auto for_loop = dyn_cast<structured_control_flow::StructuredLoop*>(&child)) {
            visit_for_impl(analysis_manager, *for_loop, undefined, open_definitions, closed_definitions);
        } else if (auto if_else = dyn_cast<structured_control_flow::IfElse*>(&child)) {
            visit_if_else_impl(analysis_manager, *if_else, undefined, open_definitions, closed_definitions);
        } else if (auto while_loop = dyn_cast<structured_control_flow::While*>(&child)) {
            visit_while_impl(analysis_manager, *while_loop, undefined, open_definitions, closed_definitions);
        } else if (auto return_statement = dyn_cast<structured_control_flow::Return*>(&child)) {
            visit_return_impl(analysis_manager, *return_statement, undefined, open_definitions, closed_definitions);
        } else if (auto sequence = dyn_cast<structured_control_flow::Sequence*>(&child)) {
            visit_sequence_impl(analysis_manager, *sequence, undefined, open_definitions, closed_definitions);
        }
    }
}

bool DataDependencyAnalysis::
    supersedes_restrictive(User& previous, User& current, analysis::AnalysisManager& analysis_manager) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    if (this->is_scalar(previous.container())) {
        return true;
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return false;
    }

    // Conservative shortcut: skip the full symbolic subset analysis and
    // assume the previous write does NOT fully cover `current`. Sound
    // (downstream consumers treat the read as potentially upward-exposed)
    // and avoids ISL queries that have caused fixed-point hangs in the
    // wider pass pipeline.
    if (!detailed_) {
        return false;
    }

    auto& assumptions_analysis = this->ensure_detailed_assumptions(analysis_manager);
    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();
    auto previous_scope = Users::scope(&previous);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    // One AssumptionsBounds per side, shared across the whole subset-pair scan.
    // The original used `previous_assumptions, previous_assumptions` (both
    // sides of `is_subset`), so we only need one bounds object here.
    symbolic::AssumptionsBounds previous_bounds(previous_assumptions);

    // Check if previous subset is subset of any current subset
    for (auto& previous_subset : previous_subsets) {
        bool found = false;
        for (auto& current_subset : current_subsets) {
            if (symbolic::is_subset(previous_subset, current_subset, previous_bounds, previous_bounds)) {
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

bool DataDependencyAnalysis::fully_covered(
    analysis::AnalysisManager& analysis_manager,
    User& current,
    const std::unordered_map<User*, std::unordered_set<User*>>* open_definitions
) {
    // open_definitions holds only writes to current's container (or is null if there are none).
    static const Definitions no_definitions;
    const Definitions& same_container = open_definitions ? *open_definitions : no_definitions;
    // Scalar reads: container-level open definition is full coverage.
    if (this->is_scalar(current.container())) {
        for (auto& w : same_container) {
            if (w.first->container() == current.container() && !this->is_undefined_user(*w.first)) {
                return true;
            }
        }
        return false;
    }
    if (this->is_undefined_user(current)) {
        return false;
    }

    // Conservative shortcut: assume coverage holds (treat the read as
    // satisfied by some open writer, equivalent to `depends`'s
    // intersect-any semantics). Sound: it allows the previous detection
    // path to drop the read into `undefined` only when truly unmatched,
    // matching pre-`fully_covered` behavior, while skipping ISL queries.
    if (!detailed_) {
        for (auto& w : same_container) {
            if (w.first->container() == current.container() && !this->is_undefined_user(*w.first)) {
                return true;
            }
        }
        return false;
    }

    auto& current_subsets = current.subsets();
    if (current_subsets.empty()) {
        // Symbol use / no real read footprint -- fall back to existence of any
        // matching open definition (depends() semantics).
        for (auto& w : same_container) {
            if (w.first->container() == current.container() && !this->is_undefined_user(*w.first)) {
                return true;
            }
        }
        return false;
    }

    auto& assumptions_analysis = this->ensure_detailed_assumptions(analysis_manager);
    auto& current_assumptions = assumptions_analysis.get(*Users::scope(&current), true);
    symbolic::AssumptionsBounds current_bounds(current_assumptions);

    // Each read subset must be contained in some single open writer's subset.
    for (auto& read_subset : current_subsets) {
        bool covered = false;
        for (auto& w_entry : same_container) {
            auto* w = w_entry.first;
            if (w->container() != current.container()) {
                continue;
            }
            if (this->is_undefined_user(*w)) {
                continue;
            }
            auto& w_assumptions = assumptions_analysis.get(*Users::scope(w), true);
            symbolic::AssumptionsBounds w_bounds(w_assumptions);
            for (auto& w_subset : w->subsets()) {
                if (symbolic::is_subset(read_subset, w_subset, current_bounds, w_bounds)) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                break;
            }
        }
        if (!covered) {
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
    if (this->is_scalar(previous.container())) {
        return true;
    }

    if (this->is_undefined_user(previous) || this->is_undefined_user(current)) {
        return true;
    }

    // Conservative shortcut: assume the two subsets may intersect. Sound
    // for dependence tracking (we may report spurious dependencies but
    // never miss a real one) and avoids ISL hangs.
    if (!detailed_) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();

    auto& assumptions_analysis = this->ensure_detailed_assumptions(analysis_manager);
    auto previous_scope = Users::scope(&previous);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    symbolic::AssumptionsBounds previous_bounds(previous_assumptions);
    symbolic::AssumptionsBounds current_bounds(current_assumptions);

    // Check if any current subset intersects with any previous subset
    bool found = false;
    for (auto& current_subset : current_subsets) {
        for (auto& previous_subset : previous_subsets) {
            if (!symbolic::is_disjoint(current_subset, previous_subset, current_bounds, previous_bounds)) {
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

    // Without detailed subset checks only scalars can be closed; skip the dominance query otherwise.
    bool scalar = this->is_scalar(previous.container());
    if (!scalar && !detailed_) {
        return false;
    }

    // Check dominance
    if (requires_dominance) {
        if (!analysis_manager.get<analysis::Users>().post_dominates(current, previous)) {
            return false;
        }
    }

    // Previous memlets are subsets of current memlets
    if (scalar) {
        return true;
    }

    // Collect memlets and assumptions
    auto& assumptions_analysis = this->ensure_detailed_assumptions(analysis_manager);
    auto previous_scope = Users::scope(&previous);
    auto current_scope = Users::scope(&current);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    symbolic::AssumptionsBounds previous_bounds(previous_assumptions);
    symbolic::AssumptionsBounds current_bounds(current_assumptions);

    auto& previous_memlets = previous.subsets();
    auto& current_memlets = current.subsets();

    for (auto& subset_ : previous_memlets) {
        bool overwritten = false;
        for (auto& subset : current_memlets) {
            if (symbolic::is_subset(subset_, subset, previous_bounds, current_bounds)) {
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
    if (this->is_scalar(previous.container())) {
        return true;
    }

    if (this->is_undefined_user(previous)) {
        return true;
    }

    // Conservative shortcut: assume `current` may depend on `previous`.
    if (!detailed_) {
        return true;
    }

    auto& assumptions_analysis = this->ensure_detailed_assumptions(analysis_manager);
    auto previous_scope = Users::scope(&previous);
    auto current_scope = Users::scope(&current);
    auto& previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto& current_assumptions = assumptions_analysis.get(*current_scope, true);

    symbolic::AssumptionsBounds previous_bounds(previous_assumptions);
    symbolic::AssumptionsBounds current_bounds(current_assumptions);

    auto& previous_memlets = previous.subsets();
    auto& current_memlets = current.subsets();

    bool intersect_any = false;
    for (auto& current_subset : current_memlets) {
        for (auto& previous_subset : previous_memlets) {
            if (!symbolic::is_disjoint(current_subset, previous_subset, current_bounds, previous_bounds)) {
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

bool DataDependencyAnalysis::is_scalar(const std::string& container) {
    auto it = this->scalar_containers_.find(container);
    if (it == this->scalar_containers_.end()) {
        bool scalar = dynamic_cast<const types::Scalar*>(&this->sdfg_.type(container)) != nullptr;
        it = this->scalar_containers_.emplace(container, scalar).first;
    }
    return it->second;
}

bool DataDependencyAnalysis::is_undefined_user(User& user) const {
    return user.owner_ == nullptr;
};

bool DataDependencyAnalysis::has_loop_boundary(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_boundaries_.find(&loop) != this->loop_boundaries_.end();
}

const std::unordered_set<User*>& DataDependencyAnalysis::
    upward_exposed_reads(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_boundaries_.at(&loop).first;
}

const std::unordered_map<User*, std::unordered_set<User*>>& DataDependencyAnalysis::
    escaping_definitions(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_boundaries_.at(&loop).second;
}

} // namespace analysis
} // namespace sdfg
