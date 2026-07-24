#include "sdfg/transformations/map_fusion.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>
#include <symengine/solve.h>
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace transformations {

class FusionConsumerSubsetVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend MapFusion;

    std::unordered_map<std::string, const data_flow::Subset*>& target_containers_;
    std::unordered_map<std::string, std::vector<data_flow::Subset>> unique_subsets_per_container_;

protected:
    bool abort() { return true; }

public:
    FusionConsumerSubsetVisitor(std::unordered_map<std::string, const data_flow::Subset*>& target_containers)
        : target_containers_(target_containers) {}

    bool visit(sdfg::structured_control_flow::Block& block) override {
        auto& dataflow = block.dataflow();
        for (auto& node : dataflow.nodes()) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
            if (access == nullptr) {
                continue;
            }
            auto& container = access->data();

            auto target_it = target_containers_.find(container);
            if (target_it == target_containers_.end()) {
                continue;
            }
            auto& producer_subset = *target_it->second;
            auto& unique_subsets = unique_subsets_per_container_[container]; // Ensures entry exists

            // Skip write-only access nodes (consumer also writes the fusion container)
            if (dataflow.in_degree(*access) > 0 && dataflow.out_degree(*access) == 0) {
                continue;
            }
            if (dataflow.in_degree(*access) != 0 || dataflow.out_degree(*access) == 0) {
                return abort();
            }

            // Check all read memlets from this access
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() != data_flow::MemletType::Computational) {
                    return abort();
                }

                auto& consumer_subset = memlet.subset();
                if (consumer_subset.size() != producer_subset.size()) {
                    return abort();
                }

                // Check if this subset is already in unique_subsets
                bool found = false;
                for (const auto& existing : unique_subsets) {
                    if (existing.size() != consumer_subset.size()) continue;
                    bool match = true;
                    for (size_t d = 0; d < existing.size(); ++d) {
                        if (!symbolic::eq(existing[d], consumer_subset[d])) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    unique_subsets.push_back(consumer_subset);
                }
            }
        }
        return false;
    }

    bool visit(sdfg::structured_control_flow::Sequence& node) override {
        for (int i = 0; i < node.size(); ++i) {
            if (dispatch(node.at(i))) {
                return true;
            }
        }

        return false;
    }

    bool visit(IfElse& node) override {
        for (int i = 0; i < node.size(); ++i) {
            if (visit(node.at(i).first)) {
                return true;
            }
        }

        return false;
    }
};

class FusionConsumerUpdateVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend MapFusion;

    builder::StructuredSDFGBuilder& builder_;
    const std::vector<MapFusion::FusionCandidate>& fusion_candidates_;
    const std::vector<std::string>& candidate_temps_;

public:
    FusionConsumerUpdateVisitor(
        builder::StructuredSDFGBuilder& builder,
        const std::vector<MapFusion::FusionCandidate>& fusion_candidates,
        const std::vector<std::string>& candidate_temps
    )
        : builder_(builder), fusion_candidates_(fusion_candidates), candidate_temps_(candidate_temps) {}

    bool dispatch_partial_sequence(Sequence& node, size_t first, size_t end) {
        for (int i = first; i < end; ++i) {
            if (dispatch(node.at(i))) {
                return true;
            }
        }

        return false;
    }

    bool visit(sdfg::structured_control_flow::Block& block) override {
        auto& dataflow = block.dataflow();

        // Snapshot access nodes before mutation: adding new access nodes below
        // would rehash dataflow.nodes_ and invalidate the range iterator.
        std::vector<data_flow::AccessNode*> access_nodes;
        for (auto& node : dataflow.nodes()) {
            auto* an = dynamic_cast<data_flow::AccessNode*>(&node);
            if (an != nullptr && dataflow.out_degree(*an) > 0) {
                access_nodes.push_back(an);
            }
        }

        for (auto* access : access_nodes) {
            std::string original_container = access->data();

            // Match each out-edge against a fusion candidate.
            struct Match {
                data_flow::Memlet* memlet;
                size_t cand_idx;
            };
            std::vector<Match> matches;
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() != data_flow::MemletType::Computational) {
                    continue;
                }
                const auto& memlet_subset = memlet.subset();
                for (size_t cand_idx = 0; cand_idx < fusion_candidates_.size(); ++cand_idx) {
                    auto& candidate = fusion_candidates_[cand_idx];
                    if (original_container != candidate.container) {
                        continue;
                    }
                    if (memlet_subset.size() != candidate.consumer_subset.size()) {
                        continue;
                    }
                    bool subset_matches = true;
                    for (size_t d = 0; d < memlet_subset.size(); ++d) {
                        if (!symbolic::eq(memlet_subset[d], candidate.consumer_subset[d])) {
                            subset_matches = false;
                            break;
                        }
                    }
                    if (subset_matches) {
                        matches.push_back({&memlet, cand_idx});
                        break;
                    }
                }
            }
            if (matches.empty()) {
                continue;
            }

            // Group matches by candidate index.
            std::unordered_set<size_t> distinct_cands;
            for (auto& m : matches) {
                distinct_cands.insert(m.cand_idx);
            }

            if (distinct_cands.size() == 1) {
                // Fast path: all matched out-edges resolve to the same candidate.
                // Mutate the shared access node in place — this preserves the
                // existing semantics for the single-read-per-container case.
                size_t cand_idx = *distinct_cands.begin();
                const auto& temp_name = candidate_temps_[cand_idx];
                auto& temp_type = builder_.subject().type(temp_name);

                access->data(temp_name);

                for (auto& m : matches) {
                    m.memlet->set_subset({});
                    m.memlet->set_base_type(temp_type);
                }

                for (auto& in_edge : dataflow.in_edges(*access)) {
                    in_edge.set_subset({});
                    in_edge.set_base_type(temp_type);
                }
            } else {
                // Stencil-like case: a single access node feeds reads at
                // multiple distinct subsets (e.g. T[j-1] and T[j+1] sharing
                // one AccessNode). Each must be rewired to its own
                // candidate-specific temp scalar — otherwise mutating
                // `access->data()` once per candidate makes all reads
                // collapse onto the last temp, e.g. T[j+1]-T[j] becomes
                // tmp-tmp == 0.
                //
                // Fix: for each distinct candidate, create one fresh
                // AccessNode for its temp scalar and redirect the matched
                // edges from the shared access node to the fresh nodes.
                struct PendingRedirect {
                    data_flow::DataFlowNode* dst;
                    std::string src_conn;
                    std::string dst_conn;
                    DebugInfo debug_info;
                    size_t cand_idx;
                    const data_flow::Memlet* memlet_to_remove;
                };
                std::vector<PendingRedirect> pending;
                pending.reserve(matches.size());
                for (auto& m : matches) {
                    pending.push_back(
                        {&m.memlet->dst(),
                         m.memlet->src_conn(),
                         m.memlet->dst_conn(),
                         m.memlet->debug_info(),
                         m.cand_idx,
                         m.memlet}
                    );
                }

                std::unordered_map<size_t, data_flow::AccessNode*> per_cand_node;
                for (auto& p : pending) {
                    auto it = per_cand_node.find(p.cand_idx);
                    if (it == per_cand_node.end()) {
                        auto& fresh = builder_.add_access(block, candidate_temps_[p.cand_idx]);
                        it = per_cand_node.emplace(p.cand_idx, &fresh).first;
                    }
                    auto& temp_type = builder_.subject().type(candidate_temps_[p.cand_idx]);
                    builder_.remove_memlet(block, *p.memlet_to_remove);
                    builder_.add_memlet(block, *it->second, p.src_conn, *p.dst, p.dst_conn, {}, temp_type, p.debug_info);
                }

                // If the original shared access node now has no edges at all
                // it is dangling and should be removed. Keep it if it still
                // has out-edges (unmatched reads of the original container)
                // or in-edges (writes to the original container).
                if (dataflow.out_degree(*access) == 0 && dataflow.in_degree(*access) == 0) {
                    builder_.remove_node(block, *access);
                }
            }
        }
        return false;
    }

    bool visit(sdfg::structured_control_flow::Sequence& node) override {
        for (int i = 0; i < node.size(); ++i) {
            if (dispatch(node.at(i))) {
                return true;
            }
        }

        return false;
    }

    bool visit(IfElse& node) override {
        for (int i = 0; i < node.size(); ++i) {
            if (visit(node.at(i).first)) {
                return true;
            }
        }

        return false;
    }
};

MapFusion::MapFusion(
    structured_control_flow::Map& first_map,
    structured_control_flow::StructuredLoop& second_loop,
    bool require_consecutive,
    bool allow_init_hoist
)
    : first_map_(first_map), second_loop_(second_loop), require_consecutive_(require_consecutive),
      allow_init_hoist_(allow_init_hoist) {}

std::string MapFusion::name() const { return "MapFusion"; }

std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> MapFusion::solve_subsets(
    const data_flow::Subset& producer_subset,
    const data_flow::Subset& consumer_subset,
    const std::vector<structured_control_flow::StructuredLoop*>& producer_loops,
    const std::vector<structured_control_flow::StructuredLoop*>& consumer_loops,
    const symbolic::Assumptions& producer_assumptions,
    const symbolic::Assumptions& consumer_assumptions,
    bool invert_range_check
) {
    // Delinearize subsets to recover multi-dimensional structure from linearized accesses
    // e.g. T[i*N + j] with assumptions on bounds -> T[i, j]
    auto producer_sub = producer_subset;
    if (producer_sub.size() == 1) {
        auto producer_result = symbolic::delinearize(producer_sub.at(0), producer_assumptions);
        if (producer_result.success) {
            producer_sub = producer_result.indices;
        }
    }
    auto consumer_sub = consumer_subset;
    if (consumer_sub.size() == 1) {
        auto consumer_result = symbolic::delinearize(consumer_sub.at(0), consumer_assumptions);
        if (consumer_result.success) {
            consumer_sub = consumer_result.indices;
        }
    }

    // Subset dimensions must match
    if (producer_sub.size() != consumer_sub.size()) {
        return {};
    }
    if (producer_sub.empty()) {
        return {};
    }

    // Extract producer indvars
    SymEngine::vec_sym producer_vars;
    for (auto* loop : producer_loops) {
        producer_vars.push_back(SymEngine::rcp_static_cast<const SymEngine::Symbol>(loop->indvar()));
    }

    // Step 1: Solve the linear equation system using SymEngine
    // System: producer_sub[d] - consumer_sub[d] = 0, for each dimension d
    // Solve for producer_vars in terms of consumer_vars and parameters
    SymEngine::vec_basic equations;
    for (size_t d = 0; d < producer_sub.size(); ++d) {
        equations.push_back(symbolic::sub(producer_sub.at(d), consumer_sub.at(d)));
    }

    // Need exactly as many equations as unknowns for a unique solution.
    // Underdetermined systems (e.g. linearized access with multiple loop vars)
    // cannot be uniquely solved and would crash linsolve.
    if (equations.size() != producer_vars.size()) {
        return {};
    }

    SymEngine::vec_basic solution;
    try {
        solution = SymEngine::linsolve(equations, producer_vars);
    } catch (...) {
        return {};
    }
    if (solution.size() != producer_vars.size()) {
        return {};
    }
    // Build consumer var set for atom validation
    symbolic::SymbolSet consumer_var_set;
    for (auto* loop : consumer_loops) {
        consumer_var_set.insert(loop->indvar());
    }

    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> mappings;
    for (size_t i = 0; i < producer_vars.size(); ++i) {
        auto& sol = solution[i];

        // Check for invalid solutions
        if (SymEngine::is_a<SymEngine::NaN>(*sol) || SymEngine::is_a<SymEngine::Infty>(*sol)) {
            return {};
        }

        // Validate that solution atoms are consumer vars or parameters
        for (const auto& atom : symbolic::atoms(sol)) {
            if (consumer_var_set.count(atom)) {
                continue;
            }
            bool is_param = false;
            auto it = consumer_assumptions.find(atom);
            if (it != consumer_assumptions.end() && it->second.constant()) {
                is_param = true;
            }
            if (!is_param) {
                it = producer_assumptions.find(atom);
                if (it != producer_assumptions.end() && it->second.constant()) {
                    is_param = true;
                }
            }
            if (!is_param) {
                return {};
            }
        }

        mappings.push_back({symbolic::symbol(producer_vars[i]->get_name()), symbolic::expand(sol)});
    }
    // Step 2: ISL integrality validation via map composition
    // Build an unconstrained producer access map (no domain bounds on producer vars).
    // In map fusion, the producer's computation is inlined into the consumer, so
    // the producer's original iteration domain is irrelevant. We only need to verify
    // that the equation system has an INTEGER solution for every consumer point.
    symbolic::Assumptions unconstrained_producer;
    for (auto* loop : producer_loops) {
        symbolic::Assumption a(loop->indvar());
        a.constant(false);
        unconstrained_producer[loop->indvar()] = a;
    }
    for (const auto& [sym, assump] : producer_assumptions) {
        if (assump.constant() && unconstrained_producer.find(sym) == unconstrained_producer.end()) {
            unconstrained_producer[sym] = assump;
        }
    }

    std::string producer_map_str = symbolic::expression_to_map_str(producer_sub, unconstrained_producer);
    // Build consumer access map with full domain constraints
    std::string consumer_map_str = symbolic::expression_to_map_str(consumer_sub, consumer_assumptions);

    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);

    isl_map* producer_map = isl_map_read_from_str(ctx, producer_map_str.c_str());
    isl_map* consumer_map = isl_map_read_from_str(ctx, consumer_map_str.c_str());

    if (!producer_map || !consumer_map) {
        if (producer_map) isl_map_free(producer_map);
        if (consumer_map) isl_map_free(consumer_map);
        isl_ctx_free(ctx);
        return {};
    }

    // Align parameters between the two maps
    isl_space* params_p = isl_space_params(isl_map_get_space(producer_map));
    isl_space* params_c = isl_space_params(isl_map_get_space(consumer_map));
    isl_space* unified = isl_space_align_params(isl_space_copy(params_p), isl_space_copy(params_c));
    isl_space_free(params_p);
    isl_space_free(params_c);

    producer_map = isl_map_align_params(producer_map, isl_space_copy(unified));
    consumer_map = isl_map_align_params(consumer_map, isl_space_copy(unified));

    // Save consumer domain before consuming consumer_map in composition
    isl_set* consumer_domain = isl_map_domain(isl_map_copy(consumer_map));

    // Compute composition: consumer_access ∘ inverse(producer_access)
    // This checks whether the equation system producer_subset = consumer_subset
    // has an integer solution for each consumer domain point.
    isl_map* producer_inverse = isl_map_reverse(producer_map);
    isl_map* composition = isl_map_apply_range(consumer_map, producer_inverse);

    // Check single-valuedness: each consumer point maps to at most one producer point
    bool single_valued = isl_map_is_single_valued(composition) == isl_bool_true;

    // Check domain coverage: every consumer point has a valid integer mapping
    isl_set* comp_domain = isl_map_domain(composition);

    bool domain_covered = isl_set_is_subset(consumer_domain, comp_domain) == isl_bool_true;

    isl_set_free(comp_domain);
    isl_set_free(consumer_domain);

    // Step 3: Verify producer write range covers consumer read range.
    // The producer only writes a subset of the array if its loops have restricted bounds.
    // Fusion is invalid if the consumer reads elements the producer never writes.
    bool range_covered = false;
    if (single_valued && domain_covered) {
        std::string constrained_producer_map_str = symbolic::expression_to_map_str(producer_sub, producer_assumptions);
        isl_map* constrained_producer = isl_map_read_from_str(ctx, constrained_producer_map_str.c_str());
        isl_map* consumer_map_copy = isl_map_read_from_str(ctx, consumer_map_str.c_str());

        if (constrained_producer && consumer_map_copy) {
            constrained_producer = isl_map_align_params(constrained_producer, isl_space_copy(unified));
            consumer_map_copy = isl_map_align_params(consumer_map_copy, isl_space_copy(unified));

            isl_set* producer_range = isl_map_range(constrained_producer);
            isl_set* consumer_range = isl_map_range(consumer_map_copy);

            // When arguments are swapped (ConsumerIntoProducer), the "producer"/"consumer"
            // labels are inverted. Flip the subset check to always verify:
            // actual_consumer_read_range ⊆ actual_producer_write_range
            if (invert_range_check) {
                range_covered = isl_set_is_subset(producer_range, consumer_range) == isl_bool_true;
            } else {
                range_covered = isl_set_is_subset(consumer_range, producer_range) == isl_bool_true;
            }

            isl_set_free(producer_range);
            isl_set_free(consumer_range);
        } else {
            if (constrained_producer) isl_map_free(constrained_producer);
            if (consumer_map_copy) isl_map_free(consumer_map_copy);
        }
    }

    isl_space_free(unified);
    isl_ctx_free(ctx);

    if (!single_valued || !domain_covered || !range_covered) {
        return {};
    }

    return mappings;
}

bool MapFusion::find_write_location(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    std::vector<structured_control_flow::StructuredLoop*>& loops,
    structured_control_flow::Sequence*& body,
    structured_control_flow::Block*& block
) {
    loops.push_back(&loop);
    auto& seq = loop.root();

    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i);

        if (auto* blk = dyn_cast<structured_control_flow::Block*>(&child)) {
            // Check if this block writes to the container
            auto& dataflow = blk->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }
                // Write access: has incoming edges (sink node)
                if (dataflow.in_degree(*access) > 0 && dataflow.out_degree(*access) == 0) {
                    if (block != nullptr) {
                        // Multiple write blocks found — ambiguous
                        return false;
                    }
                    body = &seq;
                    block = blk;
                }
            }
        } else if (auto* nested_loop = dyn_cast<structured_control_flow::StructuredLoop*>(&child)) {
            if (!find_write_location(*nested_loop, container, loops, body, block)) {
                return false;
            }
            // If we didn't find the write in this subtree, pop the loop back off
            if (loops.back() != &loop && block == nullptr) {
                // The recursive call already popped — but we need to check
            }
        }
    }

    // If we didn't find the write in this subtree, remove this loop from the chain
    if (block == nullptr) {
        loops.pop_back();
    }

    return true;
}

bool MapFusion::find_read_location(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    std::vector<structured_control_flow::StructuredLoop*>& loops,
    structured_control_flow::Sequence*& body
) {
    loops.push_back(&loop);
    auto& seq = loop.root();

    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i);

        if (auto* blk = dyn_cast<structured_control_flow::Block*>(&child)) {
            // Check if this block reads from the container
            auto& dataflow = blk->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }
                // Read access: has outgoing edges (source node)
                if (dataflow.in_degree(*access) == 0 && dataflow.out_degree(*access) > 0) {
                    if (body != nullptr && body != &seq) {
                        // Reads at different sequence levels — ambiguous
                        return false;
                    }
                    body = &seq;
                }
            }
        } else if (auto* nested_loop = dyn_cast<structured_control_flow::StructuredLoop*>(&child)) {
            if (!find_read_location(*nested_loop, container, loops, body)) {
                return false;
            }
        }
    }

    // If we didn't find any reads in this subtree, remove this loop from the chain
    if (body == nullptr) {
        loops.pop_back();
    }

    return true;
}

bool MapFusion::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    fusion_candidates_.clear();

    // no use in fusing empty loops. Also presumed to not be empty further down
    if (first_map_.root().size() == 0 || second_loop_.root().size() == 0) {
        return false;
    }

    // Criterion: Get parent scope and verify both loops are sequential children
    auto* first_parent = first_map_.get_parent();
    auto* second_parent = second_loop_.get_parent();
    if (first_parent == nullptr || second_parent == nullptr) {
        return false;
    }
    if (first_parent != second_parent) {
        return false;
    }

    auto* parent_sequence = dyn_cast<structured_control_flow::Sequence*>(first_parent);
    if (parent_sequence == nullptr) {
        return false;
    }

    int first_index = parent_sequence->index(first_map_);
    int second_index = parent_sequence->index(second_loop_);
    if (first_index == -1 || second_index == -1) {
        return false;
    }
    if (require_consecutive_ && second_index != first_index + 1) {
        return false;
    }

    // Determine fusion pattern based on nesting properties
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto first_loop_info = loop_analysis.loop_info(&first_map_);
    auto second_loop_info = loop_analysis.loop_info(&second_loop_);

    auto limit_depth = 0;

    bool first_nested = first_loop_info.is_perfectly_nested;
    bool second_nested = second_loop_info.is_perfectly_nested;

    // Both non-perfectly-nested: not supported
    if (!first_nested && !second_nested) {
        return false;
    }

    if (first_nested && second_nested) {
        // Pattern 1: Both perfectly nested — producer into consumer (original path)
        direction_ = FusionDirection::ProducerIntoConsumer;
    } else if (!first_nested && second_nested) {
        // Pattern 2: Producer non-perfectly-nested, consumer perfectly nested
        direction_ = FusionDirection::ConsumerIntoProducer;
    } else {
        // Reverse Pattern 2: Producer perfectly nested, consumer non-perfectly-nested
        direction_ = FusionDirection::ProducerIntoConsumer;
    }

    // The side being inlined must be all-parallel (all Maps) so iterations can be reordered.
    // ProducerIntoConsumer: the producer is replicated at each consumer site and must be
    // reorderable, so it must be all-parallel. The consumer is normally required to be
    // all-parallel too, because a sequential (For) loop would re-execute the inlined producer
    // on every iteration (e.g. init T=0 fused into For(k){T+=A[k]} re-initializes each k).
    //
    // Reduction branch: we relax the consumer requirement when the consumer is a perfect nest
    // (parallel outer band + inner sequential For, i.e. a reduction). A fully-parallel producer
    // that is *streamed element-by-element* inside the reduction loop can still be inlined
    // soundly (e.g. scale -> max: max(M, A[i,j,k]/d)). The element-streaming safety conditions
    // are verified once the fusion candidates are known (see consumer_reduction_branch below):
    //   (1) the fused container must not be written by the consumer (no loop-carried
    //       accumulator), and
    //   (2) its consumer read subset must depend on an inner sequential loop indvar, so the
    //       inlined producer runs once per element rather than per init position.
    // These keep init-into-reduction (T=0 followed by For(k){T+=...}) rejected.
    // ConsumerIntoProducer: only the consumer (inlined side) must be all-parallel.
    bool consumer_reduction_branch = false;
    if (direction_ == FusionDirection::ProducerIntoConsumer) {
        if (!first_loop_info.is_perfectly_parallel) {
            return false;
        } else if (!second_loop_info.is_perfectly_parallel) {
            if (!second_loop_info.is_perfectly_nested) {
                return false;
            }
            consumer_reduction_branch = true;
        }
    } else {
        if (!second_loop_info.is_perfectly_parallel) {
            return false;
        }
    }

    // Locate producer write point
    producer_loops_.clear();
    producer_body_ = nullptr;
    producer_block_ = nullptr;

    if (first_nested) {
        // Perfectly nested: walk the at(0).first chain
        producer_loops_.push_back(&first_map_);
        producer_body_ = &first_map_.root();
        structured_control_flow::ControlFlowNode* node = &first_map_.root().at(0);
        int level = 1;
        while (auto* nested = dyn_cast<structured_control_flow::StructuredLoop*>(node)) {
            if (limit_depth && ++level > limit_depth) {
                break;
            }
            producer_loops_.push_back(nested);
            producer_body_ = &nested->root();
            if (nested->root().size() == 0) return false;
            node = &nested->root().at(0);
        }
        producer_block_ = dyn_cast<structured_control_flow::Block*>(node);
        if (producer_block_ == nullptr) {
            return false;
        }
        // If the body has multiple children, the at(0) walk does not guarantee
        // we found the correct (or unique) write block. Fall back to deferred
        // find_write_location resolution.
        if (producer_body_->size() != 1) {
            producer_block_ = nullptr;
            // Keep producer_loops_ and producer_body_ from the walk — they are
            // still valid for the loop chain. find_write_location will re-resolve
            // the block within producer_body_.
        }
    } else {
        // Non-perfectly-nested: search recursively for the write block
        // We need to know which containers to look for, but we don't know them yet.
        // Defer write location search until after fusion_containers are identified.
    }

    // Locate consumer read point
    consumer_loops_.clear();
    consumer_body_ = nullptr;

    if (second_nested) {
        // Perfectly nested: walk the at(0).first chain through all loop types.
        // Reduction patterns (e.g. Map{Map{For{T[i,j]+=...}}}) are rejected by
        // the is_perfectly_parallel check — For loops make it non-parallel.
        consumer_loops_.push_back(&second_loop_);
        consumer_body_ = &second_loop_.root();
        structured_control_flow::ControlFlowNode* node = &second_loop_.root().at(0);
        int level = 1;
        while (auto* nested = dyn_cast<structured_control_flow::StructuredLoop*>(node)) {
            if (limit_depth && ++level > limit_depth) {
                break;
            }
            consumer_loops_.push_back(nested);
            consumer_body_ = &nested->root();
            if (nested->root().size() == 0) return false;
            node = &nested->root().at(0);
        }
    } else {
        // Non-perfectly-nested: defer read location search until after fusion_containers are identified.
    }

    // Get arguments analysis to identify inputs/outputs of each loop
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    auto first_args = arguments_analysis.arguments(analysis_manager, first_map_);
    auto second_args = arguments_analysis.arguments(analysis_manager, second_loop_);

    std::unordered_set<std::string> first_inputs;
    std::unordered_set<std::string> first_outputs;
    for (const auto& [name, arg] : first_args) {
        if (arg.is_output) {
            first_outputs.insert(name);
        }
        if (arg.is_input) {
            first_inputs.insert(name);
        }
    }

    std::unordered_set<std::string> second_outputs;
    for (const auto& [name, arg] : second_args) {
        if (arg.is_output) {
            second_outputs.insert(name);
        }
    }

    // First pass: identify fusion containers (producer writes, consumer reads)
    std::unordered_set<std::string> fusion_containers;
    for (const auto& [name, arg] : second_args) {
        if (first_outputs.contains(name) && arg.is_input) {
            fusion_containers.insert(name);
        }
    }
    if (fusion_containers.empty()) {
        return false;
    }

    // Second pass: check for conflicts on non-fusion containers
    for (const auto& [name, arg] : second_args) {
        bool is_fusion = fusion_containers.contains(name);
        if (first_outputs.contains(name) && arg.is_output && !is_fusion) {
            return false;
        }
        if (first_inputs.contains(name) && arg.is_output && !is_fusion) {
            return false;
        }
    }

    // Now that we know the fusion containers, resolve deferred locations
    if (producer_block_ == nullptr) {
        // Non-perfectly-nested producer (or perfectly-nested with multi-block body):
        // find write location for the first fusion container.
        // All fusion containers must be written at the same block for this to work.
        for (const auto& container : fusion_containers) {
            std::vector<structured_control_flow::StructuredLoop*> write_loops;
            structured_control_flow::Sequence* write_body = nullptr;
            structured_control_flow::Block* write_block = nullptr;

            if (!find_write_location(first_map_, container, write_loops, write_body, write_block)) {
                return false;
            }
            if (write_block == nullptr) {
                return false;
            }

            if (producer_block_ == nullptr) {
                // First container: set the locations
                producer_loops_ = write_loops;
                producer_body_ = write_body;
                producer_block_ = write_block;
            } else {
                // Subsequent containers must be in the same block
                if (write_block != producer_block_) {
                    return false;
                }
            }
        }
    }

    if (!second_nested) {
        // Non-perfectly-nested consumer: find read location for the first fusion container
        // All fusion containers must be read at the same sequence for this to work
        for (const auto& container : fusion_containers) {
            std::vector<structured_control_flow::StructuredLoop*> read_loops;
            structured_control_flow::Sequence* read_body = nullptr;

            if (!find_read_location(second_loop_, container, read_loops, read_body)) {
                return false;
            }
            if (read_body == nullptr) {
                return false;
            }

            if (consumer_body_ == nullptr) {
                // First container: set the locations
                consumer_loops_ = read_loops;
                consumer_body_ = read_body;
            } else {
                // Subsequent containers must be at the same sequence
                if (read_body != consumer_body_) {
                    return false;
                }
            }
        }
    }

    // Get assumptions for the resolved write/read locations
    // Include trivial bounds from types to help delinearization with symbolic strides
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& producer_assumptions = assumptions_analysis.get(*producer_block_, true);
    auto& consumer_assumptions = assumptions_analysis.get(consumer_body_->at(0), true);

    // Check if producer actually reads a fusion container in the dataflow.
    // If so, ProducerIntoConsumer is unsafe (original producer loop mutates the array
    // before the inlined copy reads it). Force ConsumerIntoProducer.
    // We check the dataflow directly rather than ArgumentsAnalysis, because the latter
    // conservatively marks written containers as also read.
    if (direction_ == FusionDirection::ProducerIntoConsumer) {
        auto& first_dataflow_check = producer_block_->dataflow();
        bool producer_reads_fusion = false;
        for (const auto& container : fusion_containers) {
            for (auto& node : first_dataflow_check.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access != nullptr && access->data() == container && first_dataflow_check.out_degree(*access) > 0) {
                    producer_reads_fusion = true;
                    break;
                }
            }
            if (producer_reads_fusion) break;
        }
        if (producer_reads_fusion) {
            direction_ = FusionDirection::ConsumerIntoProducer;
            // Re-check: consumer must be all-parallel for ConsumerIntoProducer
            if (!second_loop_info.is_perfectly_parallel) {
                return false;
            }
        }
    }

    // ProducerIntoConsumer only deep-copies producer_block_ into the consumer body.
    // If the producer body has multiple blocks (e.g. from prior BlockFusion merging
    // a previous fusion's writeback + inlined blocks), the write block may depend on
    // intermediates produced by earlier blocks that would NOT be copied. Reject.
    if (direction_ == FusionDirection::ProducerIntoConsumer && producer_body_->size() > 1) {
        return false;
    }

    std::unordered_map<std::string, const data_flow::Subset*> producer_subsets;

    // For each fusion container, find the producer memlet and collect unique consumer subsets
    auto& first_dataflow = producer_block_->dataflow();
    for (const auto& container : fusion_containers) {
        // Find unique producer write in first map
        data_flow::Memlet* producer_memlet = nullptr;

        for (auto& node : first_dataflow.nodes()) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
            if (access == nullptr || access->data() != container) {
                continue;
            }
            // Skip read-only access nodes (producer reads the fusion container)
            if (first_dataflow.in_degree(*access) == 0) {
                continue;
            }
            // Write access: must have exactly one incoming edge and no outgoing
            if (first_dataflow.in_degree(*access) != 1 || first_dataflow.out_degree(*access) != 0) {
                return false;
            }
            auto& iedge = *first_dataflow.in_edges(*access).begin();
            if (iedge.type() != data_flow::MemletType::Computational) {
                return false;
            }
            if (producer_memlet != nullptr) {
                return false;
            }
            producer_memlet = &iedge;
        }
        if (producer_memlet == nullptr) {
            return false;
        }

        const auto& producer_subset = producer_memlet->subset();
        if (producer_subset.empty()) {
            return false;
        } else {
            producer_subsets.emplace(container, &producer_subset);
        }
    }

    FusionConsumerSubsetVisitor consumer_visitor(producer_subsets);
    bool abort = consumer_visitor.dispatch(*consumer_body_);
    if (abort) {
        return false;
    }

    for (auto [container, unique_subsets] : consumer_visitor.unique_subsets_per_container_) {
        auto& producer_subset = *producer_subsets.at(container);
        // For each unique consumer subset, solve index mappings and create a FusionCandidate
        // The direction determines which side's indvars are solved for
        for (const auto& consumer_subset : unique_subsets) {
            std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> mappings;

            if (direction_ == FusionDirection::ProducerIntoConsumer) {
                // Solve producer indvars in terms of consumer indvars
                mappings = solve_subsets(
                    producer_subset,
                    consumer_subset,
                    producer_loops_,
                    consumer_loops_,
                    producer_assumptions,
                    consumer_assumptions
                );
            } else {
                // ConsumerIntoProducer: solve consumer indvars in terms of producer indvars
                // Arguments are swapped, so invert the range check direction
                mappings = solve_subsets(
                    consumer_subset,
                    producer_subset,
                    consumer_loops_,
                    producer_loops_,
                    consumer_assumptions,
                    producer_assumptions,
                    true
                );
            }

            if (mappings.empty()) {
                return false;
            }

            FusionCandidate candidate;
            candidate.container = container;
            candidate.consumer_subset = consumer_subset;
            candidate.index_mappings = std::move(mappings);

            fusion_candidates_.push_back(candidate);
        }
    }

    // Reduction-branch safety: when fusing a parallel producer into a non-parallel
    // (reduction) consumer, classify each fusion container into one of two sound patterns:
    //   Case 1 (stream):     the container is NOT a consumer output and its consumer read
    //                        depends on an inner sequential indvar -> it is produced and
    //                        consumed element-by-element, so the producer is scalarized and
    //                        inlined inside the reduction loop (e.g. softmax scale -> max).
    //   Case 2 (init-hoist): the container IS a consumer output (the reduction accumulator)
    //                        and its consumer read is loop-invariant w.r.t. every sequential
    //                        indvar -> the producer is the accumulator's initial value and is
    //                        hoisted to the reduction's outer parallel band, before the inner
    //                        sequential loop (e.g. T = -INF preceding T = max(T, x)).
    // Anything else (e.g. an accumulator whose read depends on the sequential indvar, or a
    // streamed value that the consumer also writes) is unsafe and rejected. The two patterns
    // require different placement in apply(), so all candidates must share one pattern.
    if (consumer_reduction_branch) {
        symbolic::SymbolSet sequential_indvars;
        size_t first_sequential = consumer_loops_.size();
        for (size_t li = 0; li < consumer_loops_.size(); ++li) {
            if (dyn_cast<structured_control_flow::Map*>(consumer_loops_[li]) == nullptr) {
                sequential_indvars.insert(consumer_loops_[li]->indvar());
                if (first_sequential == consumer_loops_.size()) {
                    first_sequential = li;
                }
            }
        }
        if (sequential_indvars.empty()) {
            return false;
        }
        bool any_stream = false;
        bool any_init = false;
        for (const auto& candidate : fusion_candidates_) {
            bool depends_on_sequential = false;
            for (const auto& dim : candidate.consumer_subset) {
                for (const auto& atom : symbolic::atoms(dim)) {
                    if (sequential_indvars.count(atom)) {
                        depends_on_sequential = true;
                        break;
                    }
                }
                if (depends_on_sequential) {
                    break;
                }
            }

            if (second_outputs.contains(candidate.container)) {
                // Case 2 candidate: must be a loop-invariant accumulator init.
                if (!allow_init_hoist_) {
                    // Init-hoisting disabled for this run (reserved for the final
                    // map-fusion pass so it does not fight loop distribution).
                    return false;
                }
                if (depends_on_sequential) {
                    return false;
                }
                any_init = true;
            } else {
                // Case 1 candidate: must be a streamed element.
                if (!depends_on_sequential) {
                    return false;
                }
                any_stream = true;
            }
        }
        // Do not mix patterns in a single fusion.
        if (any_init && any_stream) {
            return false;
        }
        if (any_init) {
            // Need an enclosing parallel band to host the hoisted init (the init must run
            // once per accumulator element, outside the sequential reduction loop).
            if (first_sequential == 0) {
                return false;
            }
            init_hoist_ = true;
            hoist_body_ = &consumer_loops_[first_sequential - 1]->root();
        }
    }

    // Criterion: At least one valid fusion candidate
    return !fusion_candidates_.empty();
}

void MapFusion::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    if (direction_ == FusionDirection::ProducerIntoConsumer) {
        // Pattern 1 + Reverse Pattern 2: Inline producer blocks into consumer's read body
        auto& first_dataflow = producer_block_->dataflow();

        // For each fusion candidate, create a temp and insert a producer block
        std::vector<std::string> candidate_temps;

        for (size_t cand_idx = 0; cand_idx < fusion_candidates_.size(); ++cand_idx) {
            auto& candidate = fusion_candidates_[cand_idx];

            auto& container_type = sdfg.type(candidate.container);
            types::Scalar tmp_type(container_type.primitive_type());
            std::string temp_name;
            if (!init_hoist_) {
                // Case 1: scalarize the streamed element into a private temp.
                temp_name = builder.find_new_name("_fused_tmp");
                builder.add_container(temp_name, tmp_type);
                candidate_temps.push_back(temp_name);
            }

            // Insert the producer block at the beginning of the host sequence:
            //  - Case 1 (stream):     consumer_body_ = innermost sequential (reduction) loop body.
            //  - Case 2 (init-hoist): hoist_body_   = outer parallel-band body, before that loop.
            auto& host_seq = init_hoist_ ? *hoist_body_ : *consumer_body_;
            auto& first_child = host_seq.at(0);
            auto& new_block = builder.add_block_before(host_seq, first_child);
            structured_control_flow::AssignmentBlock* init_assignment_block = nullptr;

            // Deep copy all nodes from producer block to new block
            std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
            std::unordered_map<std::string, std::string> intermediate_renames;
            for (auto& node : first_dataflow.nodes()) {
                node_mapping[&node] = &builder.copy_node(new_block, node);
                auto* copied = node_mapping[&node];
                if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(copied)) {
                    if (!init_hoist_ && access_node->data() == candidate.container) {
                        // Case 1: redirect the producer's array write to the private scalar.
                        access_node->data(temp_name);
                    } else if (access_node->data() == first_map_.indvar()->get_name()) {
                        // Determine the new expression for the index variable of the first map
                        symbolic::Expression new_expr = SymEngine::null;
                        for (auto& c : fusion_candidates_) {
                            for (auto& [sym, expr] : c.index_mappings) {
                                if (symbolic::eq(sym, first_map_.indvar())) {
                                    new_expr = expr;
                                    break;
                                }
                            }
                            if (!new_expr.is_null()) {
                                break;
                            }
                        }

                        if (new_expr.is_null() || symbolic::eq(new_expr, second_loop_.indvar())) {
                            // Simple case: The new expression is simply the index variable of the second loop
                            access_node->data(second_loop_.indvar()->get_name());
                        } else {
                            // Complex case: Add AssignmentBlock before the new block (if necessary) and store the
                            // shifted index into a new temporary variable with an assignment. Then, replace the index
                            // variable with the new temporary variable
                            auto new_index_name = builder.find_new_name();
                            builder
                                .add_container(new_index_name, builder.subject().type(second_loop_.indvar()->get_name()));

                            if (!init_assignment_block) {
                                init_assignment_block = &builder.add_assignments_at(host_seq, 0, {});
                            }
                            init_assignment_block->assignments().insert({symbolic::symbol(new_index_name), new_expr});
                            access_node->data(new_index_name);
                        }
                    } else if (first_dataflow.in_degree(node) > 0 && first_dataflow.out_degree(node) > 0 &&
                               dynamic_cast<const types::Scalar*>(&sdfg.type(access_node->data())) != nullptr) {
                        // SSA Dataflow required to check for non-local use of the access node's container.
                        // Intermediate access node (e.g. from a prior BlockFusion): clone
                        // its container so each inlined copy gets its own private scalar
                        auto it = intermediate_renames.find(access_node->data());
                        if (it == intermediate_renames.end()) {
                            std::string fresh = builder.find_new_name(access_node->data());
                            builder.add_container(fresh, sdfg.type(access_node->data()));
                            intermediate_renames[access_node->data()] = fresh;
                        }
                        access_node->data(intermediate_renames[access_node->data()]);
                    }
                }
            }

            // Add memlets with index substitution (producer indvars → consumer expressions)
            for (auto& edge : first_dataflow.edges()) {
                auto& src_node = edge.src();
                auto& dst_node = edge.dst();

                const types::IType* base_type = &edge.base_type();
                data_flow::Subset new_subset;
                for (const auto& dim : edge.subset()) {
                    auto new_dim = dim;
                    for (const auto& [pvar, mapping] : candidate.index_mappings) {
                        new_dim = symbolic::subs(new_dim, pvar, mapping);
                    }
                    new_dim = symbolic::expand(new_dim);
                    new_subset.push_back(new_dim);
                }

                // Case 1: the producer's array write becomes a scalar write (empty subset).
                // Case 2: keep the remapped array subset so the init writes the accumulator.
                auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
                if (!init_hoist_ && dst_access != nullptr && dst_access->data() == candidate.container &&
                    first_dataflow.in_degree(*dst_access) > 0) {
                    new_subset.clear();
                    base_type = &tmp_type;
                }

                builder.add_memlet(
                    new_block,
                    *node_mapping[&src_node],
                    edge.src_conn(),
                    *node_mapping[&dst_node],
                    edge.dst_conn(),
                    new_subset,
                    *base_type,
                    edge.debug_info()
                );
            }
        }

        // Case 1 only: rewrite consumer reads of the fused arrays to the scalar temps.
        // Case 2 leaves the reduction body untouched (it keeps reading/writing the accumulator,
        // now pre-initialized by the hoisted init block).
        if (!init_hoist_) {
            size_t num_producer_blocks = fusion_candidates_.size();
            FusionConsumerUpdateVisitor update_visitor(builder, fusion_candidates_, candidate_temps);
            update_visitor.dispatch_partial_sequence(*consumer_body_, num_producer_blocks, consumer_body_->size());
        } else {
            // Case 2: the hoisted init copy fully overwrites the accumulator before the
            // reduction reads it, so the original init producer map is redundant. Unlike
            // Case 1, the accumulator array stays live, so DCE would not reclaim it — remove
            // the producer explicitly (mirrors how ConsumerIntoProducer removes its loop).
            auto* parent = first_map_.get_parent();
            auto* parent_seq = dyn_cast<structured_control_flow::Sequence*>(parent);
            if (parent_seq != nullptr) {
                int idx = parent_seq->index(first_map_);
                if (idx >= 0) {
                    builder.remove_child(*parent_seq, static_cast<size_t>(idx));
                }
            }
        }

    } else {
        // ConsumerIntoProducer (Pattern 2): Inline consumer blocks into the producer's write body
        // Modify the producer block in-place to write to a temp scalar, add a writeback block
        // for the original array, then copy consumer blocks reading from the temp.

        std::vector<std::string> candidate_temps;
        auto& producer_dataflow = producer_block_->dataflow();

        for (size_t cand_idx = 0; cand_idx < fusion_candidates_.size(); ++cand_idx) {
            auto& candidate = fusion_candidates_[cand_idx];

            auto& container_type = sdfg.type(candidate.container);
            std::string temp_name = builder.find_new_name("_fused_tmp");
            types::Scalar tmp_type(container_type.primitive_type());
            builder.add_container(temp_name, tmp_type);
            candidate_temps.push_back(temp_name);

            // Step 1: Modify the original producer block to write to _fused_tmp
            data_flow::Subset original_write_subset;
            for (auto& node : producer_dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != candidate.container) continue;
                if (producer_dataflow.in_degree(*access) == 0) continue;

                // This is the write access node — save the original subset, then redirect
                for (auto& in_edge : producer_dataflow.in_edges(*access)) {
                    original_write_subset = in_edge.subset();
                    in_edge.set_subset({});
                    in_edge.set_base_type(tmp_type);
                }
                access->data(temp_name);
                break;
            }

            // Step 2: Add a writeback block: container[original_subset] = _fused_tmp
            auto& wb_block = builder.add_block_after(*producer_body_, *producer_block_);
            auto& wb_src = builder.add_access(wb_block, temp_name);
            auto& wb_dst = builder.add_access(wb_block, candidate.container);
            auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {});
            builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, original_write_subset);

            // Step 3: Copy consumer blocks after the writeback block
            structured_control_flow::ControlFlowNode* last_inserted = &wb_block;

            for (size_t i = 0; i < consumer_body_->size(); ++i) {
                auto* consumer_block = dyn_cast<structured_control_flow::Block*>(&consumer_body_->at(i));
                if (consumer_block == nullptr) {
                    continue;
                }

                auto& consumer_dataflow = consumer_block->dataflow();

                // Check if this block reads from the fusion container
                bool reads_container = false;
                for (auto& node : consumer_dataflow.nodes()) {
                    auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                    if (access != nullptr && access->data() == candidate.container &&
                        consumer_dataflow.out_degree(*access) > 0) {
                        reads_container = true;
                        break;
                    }
                }
                if (!reads_container) {
                    continue;
                }

                // Insert a new block after the last inserted block in the producer's body
                auto& new_block = builder.add_block_after(*producer_body_, *last_inserted);
                structured_control_flow::AssignmentBlock* init_assignment_block = nullptr;

                // Deep copy all nodes from consumer block
                std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
                std::unordered_map<std::string, std::string> intermediate_renames;
                for (auto& node : consumer_dataflow.nodes()) {
                    node_mapping[&node] = &builder.copy_node(new_block, node);
                    auto* copied = node_mapping[&node];
                    if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(copied)) {
                        if (access_node->data() == candidate.container) {
                            // Only rename read access nodes to temp; keep write access nodes
                            // pointing to the original container
                            if (consumer_dataflow.in_degree(node) == 0) {
                                access_node->data(temp_name);
                            }
                        } else if (consumer_dataflow.in_degree(node) > 0 && consumer_dataflow.out_degree(node) > 0 &&
                                   dynamic_cast<const types::Scalar*>(&sdfg.type(access_node->data())) != nullptr) {
                            // SSA Dataflow required to check for non-local use of the access node's container.
                            // Intermediate access node (e.g. from a prior BlockFusion): clone
                            // its container so each inlined copy gets its own private scalar
                            auto it = intermediate_renames.find(access_node->data());
                            if (it == intermediate_renames.end()) {
                                std::string fresh = builder.find_new_name(access_node->data());
                                builder.add_container(fresh, sdfg.type(access_node->data()));
                                intermediate_renames[access_node->data()] = fresh;
                            }
                            access_node->data(intermediate_renames[access_node->data()]);
                        }
                        if (access_node->data() == second_loop_.indvar()->get_name() &&
                            consumer_dataflow.in_degree(node) == 0) {
                            // Determine the new expression for the index variable of the second loop
                            symbolic::Expression new_expr = SymEngine::null;
                            for (auto& c : fusion_candidates_) {
                                for (auto& [sym, expr] : c.index_mappings) {
                                    if (symbolic::eq(sym, second_loop_.indvar())) {
                                        new_expr = expr;
                                        break;
                                    }
                                }
                                if (!new_expr.is_null()) {
                                    break;
                                }
                            }

                            if (new_expr.is_null() || symbolic::eq(new_expr, first_map_.indvar())) {
                                // Simple case: The new expression is simply the index variable of the first map
                                access_node->data(first_map_.indvar()->get_name());
                            } else {
                                // Complex case: Add an AssignmentBlock (if necessary) and store the
                                // shifted index into a new temporary variable with an assignment. Then, replace the
                                // index variable with the new temporary variable
                                if (!init_assignment_block) {
                                    init_assignment_block = &builder.add_assignments_at(*producer_body_, 0, {});
                                }
                                auto new_index_name = builder.find_new_name();
                                builder.add_container(
                                    new_index_name, builder.subject().type(first_map_.indvar()->get_name())
                                );
                                init_assignment_block->assignments().insert({symbolic::symbol(new_index_name), new_expr}
                                );
                                access_node->data(new_index_name);
                            }
                        }
                    }
                }

                // Add memlets with index substitution (consumer indvars → producer expressions)
                for (auto& edge : consumer_dataflow.edges()) {
                    auto& src_node = edge.src();
                    auto& dst_node = edge.dst();

                    const types::IType* base_type = &edge.base_type();
                    data_flow::Subset new_subset;
                    for (const auto& dim : edge.subset()) {
                        auto new_dim = dim;
                        for (const auto& [cvar, mapping] : candidate.index_mappings) {
                            new_dim = symbolic::subs(new_dim, cvar, mapping);
                        }
                        new_dim = symbolic::expand(new_dim);
                        new_subset.push_back(new_dim);
                    }

                    // For read edges from temp scalar, use empty subset
                    auto* src_access = dynamic_cast<data_flow::AccessNode*>(&src_node);
                    if (src_access != nullptr && src_access->data() == candidate.container &&
                        consumer_dataflow.in_degree(*src_access) == 0) {
                        new_subset.clear();
                        base_type = &tmp_type;
                    }

                    builder.add_memlet(
                        new_block,
                        *node_mapping[&src_node],
                        edge.src_conn(),
                        *node_mapping[&dst_node],
                        edge.dst_conn(),
                        new_subset,
                        *base_type,
                        edge.debug_info()
                    );
                }

                last_inserted = &new_block;
            }
        }

        // Remove the consumer loop
        auto* parent = second_loop_.get_parent();
        auto* parent_seq = dyn_cast<structured_control_flow::Sequence*>(parent);
        if (parent_seq != nullptr) {
            int idx = parent_seq->index(second_loop_);
            if (idx >= 0) {
                builder.remove_child(*parent_seq, static_cast<size_t>(idx));
            }
        }
    }

    analysis_manager.invalidate_all();
    applied_ = true;
}

void MapFusion::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], first_map_);

    j["subgraph"]["1"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["1"], second_loop_);
}

MapFusion MapFusion::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto first_map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto second_loop_id = desc["subgraph"]["1"]["element_id"].get<size_t>();

    auto first_element = builder.find_element_by_id(first_map_id);
    auto second_element = builder.find_element_by_id(second_loop_id);

    if (first_element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(first_map_id) + " not found.");
    }
    if (second_element == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_loop_id) + " not found."
        );
    }

    auto* first_map = dyn_cast<structured_control_flow::Map*>(first_element);
    auto* second_loop = dyn_cast<structured_control_flow::StructuredLoop*>(second_element);

    if (first_map == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(first_map_id) + " is not a Map."
        );
    }
    if (second_loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_loop_id) + " is not a StructuredLoop."
        );
    }

    return MapFusion(*first_map, *second_loop);
}

} // namespace transformations
} // namespace sdfg
