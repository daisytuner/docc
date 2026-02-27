#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"

namespace sdfg {
namespace passes {

DeadDataElimination::DeadDataElimination() : Pass(), permissive_(false) {};

DeadDataElimination::DeadDataElimination(bool permissive) : Pass(), permissive_(permissive) {};

std::string DeadDataElimination::name() { return "DeadDataElimination"; };

bool DeadDataElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

    // Eliminate dead code, i.e., never read
    std::unordered_set<std::string> dead;
    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (users.num_views(name) > 0 || users.num_moves(name) > 0) {
            continue;
        }
        bool no_reads = users.num_reads(name) == 0;
        if (no_reads && users.num_writes(name) == 0) {
            dead.insert(name);
            applied = true;
            continue;
        }

        // TODO UNSAFE: use analysis does not return actual reads and writes for pointers. So if [name] is a pointer,
        // no_reads, does not actually mean no reads exist and any removal is problematic
        // but fixing may have vast downstream effects that not allowed at this time

        bool completely_unused = no_reads; // if there are reads left, we can never remove the container, but maybe some
                                           // writes
        auto raws = data_dependency_analysis.definitions(name);
        for (auto set : raws) {
            bool no_reads = false;
            if (set.second.size() == 0) {
                no_reads = true;
            }
            if (data_dependency_analysis.is_undefined_user(*set.first)) {
                continue;
            }

            if (no_reads) {
                bool could_eliminate_write = false;
                auto write = set.first;
                if (auto transition = dynamic_cast<structured_control_flow::Transition*>(write->element())) {
                    transition->assignments().erase(symbolic::symbol(name));
                    applied = true;
                    could_eliminate_write = true;
                } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(write->element())) {
                    auto& graph = access_node->get_parent();

                    auto& src = (*graph.in_edges(*access_node).begin()).src();
                    if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&src)) {
                        auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                        builder.clear_node(block, *tasklet);
                        applied = true;
                        could_eliminate_write = true;
                    } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(&src)) {
                        if (!library_node->side_effect() ||
                            (permissive_ && library_node->code() == stdlib::LibraryNodeType_Malloc)) {
                            auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                            builder.clear_node(block, *library_node);
                            applied = true;
                            could_eliminate_write = true;
                        }
                    }
                }

                completely_unused &= could_eliminate_write;
            }
        }

        if (completely_unused) { // no reads, and all remaining writes could be removed
            dead.insert(name);
        }
    }

    for (auto& name : dead) {
        builder.remove_container(name);
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
