#include "sdfg/passes/dataflow/trivial_array_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

TrivialArrayElimination::TrivialArrayElimination()
    : Pass() {

      };

std::string TrivialArrayElimination::name() {
    return "TrivialArrayElimination";
};

bool TrivialArrayElimination::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    for (auto& name : sdfg.containers()) {
        // Criterion: Only transients
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (users.num_moves(name) > 0 || users.num_views(name) > 0) {
            continue;
        }

        // Criterion: Data type must be or contain arrays
        std::unique_ptr<types::IType> type = sdfg.type(name).clone();
        uint depth = 0;
        bool has_trivial_dimension = false;
        while (auto atype = dynamic_cast<const types::Array*>(type.get())) {
            if (symbolic::eq(atype->num_elements(), symbolic::one())) {
                has_trivial_dimension = true;
                break;
            }
            type = atype->element_type().clone();
            depth++;
        }

        if (!has_trivial_dimension) {
            continue;
        }

        // Construct new type
        auto atype = static_cast<const types::Array*>(type.get());
        std::unique_ptr<types::IType> inner_type = atype->element_type().clone();
        std::unique_ptr<types::IType> new_type = types::recombine_array_type(sdfg.type(name), depth, *inner_type.get());

        // Replace data type
        builder.change_type(name, *new_type.get());

        // Replace all reads
        for (auto& user : users.reads(name)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            assert(access_node != nullptr && "Expected AccessNode");

            auto& graph = access_node->get_parent();
            for (auto& oedge : graph.out_edges(*access_node)) {
                auto subset = oedge.subset();
                subset.erase(subset.begin() + depth);
                oedge.set_subset(subset);
                oedge.set_base_type(*new_type);
            }
        }
        // Replace all writes
        for (auto& user : users.writes(name)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            assert(access_node != nullptr && "Expected AccessNode");

            auto& graph = access_node->get_parent();
            for (auto& iedge : graph.in_edges(*access_node)) {
                auto subset = iedge.subset();
                subset.erase(subset.begin() + depth);
                iedge.set_subset(subset);
                iedge.set_base_type(*new_type);
            }
        }

        applied = true;
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
