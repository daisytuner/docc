#include "sdfg/passes/redundant_load_elimination_pass.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "symengine/subs.h"

namespace sdfg::passes {

RedundantLoadVisitor::
    RedundantLoadVisitor(builder::StructuredSDFGBuilder& builder, RedundantLoadEliminationPass::State& state)
    : builder_(builder), state_(state) {}

bool RedundantLoadVisitor::visit(sdfg::structured_control_flow::Block& block) {
    auto& dflow = block.dataflow();
    for (auto* access_node : dflow.data_nodes()) {
        if (dynamic_cast<data_flow::ConstantNode*>(access_node) == nullptr) {
            auto* in_edge = dflow.in_edge(*access_node);
            if (!in_edge) {
                continue;
            }
            auto* src_node = dynamic_cast<const data_flow::CodeNode*>(&in_edge->src());
            if (!src_node) {
                continue;
            }
            auto& type_to_match = in_edge->base_type();
            auto& subset_to_match = in_edge->subset();
            if (subset_to_match.empty()) {
                continue;
            }

            auto out_edges = dflow.out_edges(*access_node);

            bool matches = true;
            for (auto& edge : out_edges) {
                if (edge.base_type() != type_to_match) {
                    matches = false;
                    break;
                }

                if (!symbolic::vectors_of_expressions_match(edge.subset(), subset_to_match)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                // redirect the write to a scalar temp. Then write into original access_node (indirect)
                // replace all reads of the same index with reads of the temp
                auto bypass_name = builder_.find_new_name();
                auto bypass_type = in_edge->result_type(builder_.subject());
                builder_.add_container(bypass_name, *bypass_type);
                auto& bypass_access = builder_.add_access(block, bypass_name);
                builder_.add_computational_memlet(
                    block,
                    *const_cast<data_flow::CodeNode*>(src_node),
                    in_edge->src_conn(),
                    bypass_access,
                    {},
                    *bypass_type,
                    {}
                );
                auto& copy_tasklet = builder_.add_tasklet(block, data_flow::assign, "out", {"in"});
                builder_.add_computational_memlet(block, bypass_access, copy_tasklet, "in", {}, *bypass_type);
                builder_.add_computational_memlet(
                    block, copy_tasklet, "out", *access_node, subset_to_match, type_to_match, in_edge->debug_info()
                );

                std::vector<data_flow::Memlet*> to_remove;
                for (auto& replace : out_edges) {
                    to_remove.push_back(&replace);
                    builder_.add_memlet(
                        block,
                        bypass_access,
                        "void",
                        replace.dst(),
                        replace.dst_conn(),
                        {},
                        *bypass_type,
                        replace.debug_info()
                    );
                }

                builder_.remove_memlet(block, *in_edge);
                in_edge = nullptr;
                for (auto* memlet : to_remove) {
                    builder_.remove_memlet(block, *memlet);
                }

                state_.optimized++;
            }
        }
    }
    return true;
}

bool RedundantLoadEliminationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    State state;

    RedundantLoadVisitor v(builder, state);
    v.dispatch(builder.subject().root());

    return state.optimized > 0;
}

} // namespace sdfg::passes
