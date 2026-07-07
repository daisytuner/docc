#include "sdfg/passes/offloading/remove_redundant_transfers_pass.h"

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/offloading/data_offloading_node.h"


namespace sdfg {
namespace passes {

RemoveRedundantTransfersPass::RemoveRedundantTransfersPass() : Pass() {}


std::string RemoveRedundantTransfersPass::name() { return "RemoveRedundantTransfersPass"; };

bool RemoveRedundantTransfersPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& users = analysis_manager.get<analysis::Users>();
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();

    std::unordered_set<offloading::DataOffloadingNode*> redundant_transfers;

    for (auto write : users.writes()) {
        if (auto* access = dynamic_cast<data_flow::AccessNode*>(write->element())) {
            int in_degree = access->get_parent().in_degree(*access);
            if (in_degree != 1) {
                continue;
            }

            auto in_edges = access->get_parent().in_edges(*access);
            auto& in_edge = *in_edges.begin();
            if (auto* data_transfer = dynamic_cast<offloading::DataOffloadingNode*>(&in_edge.src())) {
                if (!data_transfer->is_d2h()) {
                    continue;
                }
                auto container = write->container();

                for (auto write2 : users.writes(container)) {
                    if (write == write2) {
                        continue;
                    }
                    if (!(dominance_analysis.dominates(*write, *write2) &&
                          dominance_analysis.post_dominates(*write2, *write))) {
                        continue;
                    }

                    if (auto* access2 = dynamic_cast<data_flow::AccessNode*>(write2->element())) {
                        int in_degree = access2->get_parent().in_degree(*access2);
                        if (in_degree != 1) {
                            continue;
                        }

                        auto in_edges = access2->get_parent().in_edges(*access2);
                        auto& in_edge = *in_edges.begin();

                        if (auto* data_transfer2 = dynamic_cast<offloading::DataOffloadingNode*>(&in_edge.src())) {
                            if (!data_transfer2->is_d2h()) {
                                continue;
                            }
                            if (data_transfer != data_transfer2 && data_transfer->equal_with(*data_transfer2)) {
                                auto all_users_between = users.all_uses_between(*write, *write2);
                                std::unordered_set<analysis::User*> relevant_uses_between;
                                for (auto user : all_users_between) {
                                    if (user->container() == container) {
                                        relevant_uses_between.insert(user);
                                        break;
                                    }
                                }
                                if (relevant_uses_between.empty()) {
                                    redundant_transfers.insert(data_transfer);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto transfer : redundant_transfers) {
        if (auto block = dyn_cast<structured_control_flow::Block*>(transfer->get_parent().get_parent())) {
            builder.clear_code_node_legacy(*block, *transfer);
            applied = true;
        }
    }

    return applied;
};


} // namespace passes
} // namespace sdfg
