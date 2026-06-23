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

struct RedundantCandidate {
    data_flow::AccessNode* access_node;
    const data_flow::Memlet* in_edge;
    std::vector<data_flow::Memlet*> out_edges;
    bool write_redundant = false;
    bool too_complex = false;
};

bool RedundantLoadVisitor::visit(sdfg::structured_control_flow::Block& block) {
    auto& dflow = block.dataflow();
    auto topo = dflow.topological_sort();

    std::vector<std::unique_ptr<RedundantCandidate>> candidates;
    std::unordered_map<std::string, RedundantCandidate*> by_container;

    for (auto* node : topo) {
        if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (dynamic_cast<data_flow::ConstantNode*>(access_node) == nullptr) {
                auto it = by_container.find(access_node->data());
                RedundantCandidate* existing_candidate = nullptr;
                if (it != by_container.end()) {
                    existing_candidate = it->second;
                }

                const data_flow::Subset* write_subset = nullptr;
                const types::IType* write_type = nullptr;
                auto* in_edge = dflow.in_edge_if_single(*access_node); // not supported for multiple input edges
                if (in_edge && in_edge->is_dst_pointed_to_write()) {
                    auto* src_node = dynamic_cast<const data_flow::CodeNode*>(&in_edge->src());
                    if (src_node) {
                        write_subset = &in_edge->subset();
                        write_type = &in_edge->base_type();
                    }
                }

                if (existing_candidate && write_subset) {
                    if (symbolic::vectors_of_expressions_match(*write_subset, existing_candidate->in_edge->subset())) {
                        existing_candidate->write_redundant = true;
                        continue;
                    } else {
                        existing_candidate->too_complex = true;
                        continue;
                    }
                } else if (!existing_candidate && write_subset) {
                    auto out_edges = dflow.out_edges(*access_node);

                    bool any_reads = false;
                    bool reads_match_write_edge = true;
                    for (auto& edge : out_edges) {
                        any_reads = true;
                        if (edge.base_type() != *write_type || !edge.is_src_pointed_to_read()) {
                            reads_match_write_edge = false;
                            break;
                        }

                        if (!symbolic::vectors_of_expressions_match(edge.subset(), *write_subset)) {
                            reads_match_write_edge = false;
                            break;
                        }
                    }

                    if (any_reads && reads_match_write_edge) {
                        auto& candidate = candidates.emplace_back(std::make_unique<RedundantCandidate>());
                        candidate->access_node = access_node;
                        candidate->in_edge = in_edge;
                        std::ranges::transform(out_edges, std::back_inserter(candidate->out_edges), [](auto& edge) {
                            return &edge;
                        });
                        by_container[access_node->data()] = candidate.get();
                    }
                }
            }
        }
    }

    for (auto& candidate : candidates) {
        if (!candidate->too_complex) {
            // redirect the write to a scalar temp. Then write into original access_node (indirect)
            // replace all reads of the same index with reads of the temp
            auto& access_node = *candidate->access_node;
            auto& in_edge = *candidate->in_edge;
            auto& out_edges = candidate->out_edges;

            auto bypass_name = builder_.find_new_name("rle_");
            auto bypass_type = in_edge.result_type(builder_.subject());
            builder_.add_container(bypass_name, *bypass_type);
            auto& bypass_access = builder_.add_access(block, bypass_name);
            builder_.add_computational_memlet(
                block,
                const_cast<data_flow::CodeNode&>(static_cast<const data_flow::CodeNode&>(in_edge.src())),
                in_edge.src_conn(),
                bypass_access,
                {},
                *bypass_type,
                {}
            );
            if (!candidate->write_redundant) {
                auto& copy_tasklet = builder_.add_tasklet(block, data_flow::assign, "out", {"in"});
                builder_.add_computational_memlet(block, bypass_access, copy_tasklet, "in", {}, *bypass_type);
                builder_.add_computational_memlet(
                    block,
                    copy_tasklet,
                    "out",
                    *candidate->access_node,
                    in_edge.subset(),
                    in_edge.base_type(),
                    in_edge.debug_info()
                );
            }

            for (auto* replace : out_edges) {
                builder_.add_memlet(
                    block,
                    bypass_access,
                    "void",
                    replace->dst(),
                    replace->dst_conn(),
                    {},
                    *bypass_type,
                    replace->debug_info()
                );
            }

            builder_.remove_memlet(block, in_edge);
            for (auto* memlet : out_edges) {
                builder_.remove_memlet(block, *memlet);
            }

            state_.optimized++;
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
