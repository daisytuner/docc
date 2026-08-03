#include "sdfg/passes/loop_fusion/loop_fusion_by_accesses.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/passes/loop_fusion/loop_fusion_pass.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/transformations/map_fusion.h"
#include "symengine/solve.h"

namespace sdfg::passes::loop_fusion {

structured_control_flow::StructuredLoop* LoopFusionByAccessWorker::Plan::consumer_target_loop() const {
    if (this->init_hoist_) {
        return this->consumer_loops_.at(this->consumer_loops_.size() - 2);
    } else {
        return this->consumer_loops_.back();
    }
}

structured_control_flow::Sequence& LoopFusionByAccessWorker::Plan::consumer_target_sequence() const {
    return init_hoist_ ? *hoist_body_ : *consumer_body_;
}

std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> LoopFusionByAccessWorker::solve_subsets(
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
        auto equation = symbolic::sub(producer_sub.at(d), consumer_sub.at(d));
        if (!symbolic::eq(equation, symbolic::zero())) {
            equations.push_back(equation);
        }
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

LoopFusionByAccessWorker::FusionRegs LoopFusionByAccessWorker::
    find_fusion_regs(const FusionLoopCandidate& first, const FusionLoopCandidate& second) {
    auto& first_args = first.args;
    auto second_args = second.args;

    std::unordered_set<std::string> first_inputs;
    std::unordered_set<std::string> first_outputs;
    for (const auto& [name, arg] : first_args) {
        if (arg.arg.is_output) {
            first_outputs.insert(name);
        }
        if (arg.arg.is_input) {
            first_inputs.insert(name);
        }
    }

    std::unordered_set<std::string> second_outputs;
    for (const auto& [name, arg] : second_args) {
        if (arg.arg.is_output) {
            second_outputs.insert(name);
        }
    }

    // First pass: identify fusion containers (producer writes, consumer reads)
    std::unordered_set<std::string> fusion_containers;
    for (const auto& [name, arg] : second_args) {
        if (first_outputs.contains(name) && arg.arg.is_input) {
            fusion_containers.insert(name);
        }
    }

    // Second pass: check for conflicts on non-fusion containers
    for (const auto& [name, arg] : second_args) {
        bool is_fusion = fusion_containers.contains(name);
        if (first_outputs.contains(name) && arg.arg.is_output && !is_fusion) {
            return {.conflicts = true};
        }
        if (first_inputs.contains(name) && arg.arg.is_output && !is_fusion) {
            return {.conflicts = true};
        }
    }

    return {.fusion_regs = fusion_containers, .second_outputs = second_outputs, .conflicts = false};
}

std::vector<StructuredLoop*> LoopFusionByAccessWorker::collect_structured_sub_tree(StructuredLoop& top) {
    auto& ana = get_loop_analysis();
    std::vector<StructuredLoop*> loop_stack;
    auto it = ana.get_loop_iterator(&top);
    auto end = ana.get_subtree_end(&top);
    auto size = std::distance(it, end);
    loop_stack.reserve(size);
    while (it != end) {
        auto* loop = *it;
        if (auto structured = dyn_cast<StructuredLoop*>(loop)) {
            loop_stack.push_back(structured);
        } else {
            return {};
        }
        ++it;
    }
    return loop_stack;
}

std::vector<StructuredLoop*> LoopFusionByAccessWorker::collect_loop_parents(Sequence* sequence, StructuredLoop* loop) {
    auto node = sequence->get_parent();
    std::vector<StructuredLoop*> loop_stack;
    while (node != loop) {
        if (node == nullptr) {
            throw std::runtime_error(
                "The loop #" + std::to_string(loop->element_id()) + " must be parent of #" +
                std::to_string(sequence->element_id())
            );
        }
        if (auto structured = dyn_cast<StructuredLoop*>(node)) {
            loop_stack.push_back(structured);
        }
        node = node->get_parent();
    }
    loop_stack.push_back(loop);

    std::ranges::reverse(loop_stack);

    return loop_stack;
}

std::unique_ptr<LoopFusionByAccessWorker::Plan> LoopFusionByAccessWorker::
    try_create_fusion_by_access_plan(FusionLoopCandidate& first, FusionLoopCandidate& second, bool domains_match) {
    auto first_map = dyn_cast<Map*>(first.loop);
    if (!first_map) {
        return {};
    }
    auto state_ptr = std::make_unique<Plan>(*first_map, *second.loop);
    Plan& state = *state_ptr;

    auto& ana = get_loop_analysis();
    auto first_loop_info = ana.loop_info(first.loop);
    auto second_loop_info = ana.loop_info(second.loop);

    bool first_nested = first_loop_info.is_perfectly_nested;
    bool second_nested = second_loop_info.is_perfectly_nested;

    // Both non-perfectly-nested: not supported
    if (!first_nested && !second_nested) {
        return {};
    }

    if (!first_nested && second_nested) {
        // Pattern 2: Producer non-perfectly-nested, consumer perfectly nested
        state.direction_ = FusionDirection::ConsumerIntoProducer;
        DEBUG_PRINTLN(
            "Aborting ConsumerIntoProducer still unsupported: #" + std::to_string(first.loop->element_id()) + " <- #" +
            std::to_string(second.loop->element_id())
        );
        return {}; // unsupported for now. to different
    } else {
        // Pattern 1: Both perfectly nested — producer into consumer (original path)
        // Reverse Pattern 2: Producer perfectly nested, consumer non-perfectly-nested
        state.direction_ = FusionDirection::ProducerIntoConsumer;
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
    if (state.direction_ == FusionDirection::ProducerIntoConsumer) {
        if (!first_loop_info.is_perfectly_parallel) {
            return {};
        } else if (!second_loop_info.is_perfectly_parallel) {
            if (!second_loop_info.is_perfectly_nested) {
                return {};
            }
            consumer_reduction_branch = true;
        }
    } else {
        if (!second_loop_info.is_perfectly_parallel) {
            return {};
        }
    }

    // Locate producer write point

    if (first_nested) {
        // perfectly nested
        state.producer_loops_ = collect_structured_sub_tree(state.first);
        if (state.producer_loops_.empty()) {
            // sth. is invalid here, it should have been perfectly nested and at least contain the loop itself
            return {};
        }
        state.producer_body_ = &state.producer_loops_.back()->root();
        if (state.producer_body_->size() == 0) {
            return {};
        }

        if (state.producer_body_->size() == 1) {
            structured_control_flow::ControlFlowNode* node = &state.producer_body_->at(0);
            state.producer_block_ = dyn_cast<structured_control_flow::Block*>(node);
            if (state.producer_block_ == nullptr) {
                return {};
            }
        }
        // If the body has multiple children, then rely on write-based identification
    } else {
        // Non-perfectly-nested: search recursively for the write block
        // We need to know which containers to look for, but we don't know them yet.
        // Defer write location search until after fusion_containers are identified.
    }

    // Locate consumer read point
    if (second_nested) {
        // Perfectly nested: subtree should just be a stack of loops
        // Reduction patterns (e.g. Map{Map{For{T[i,j]+=...}}}) are rejected by
        // the is_perfectly_parallel check — For loops make it non-parallel.
        state.consumer_loops_ = collect_structured_sub_tree(state.second);
        state.consumer_body_ = &state.consumer_loops_.back()->root();
    } else {
        // Non-perfectly-nested: defer read location search until after fusion_containers are identified.
    }

    // Get arguments analysis to identify inputs/outputs of each loop
    auto [fusion_regs, second_outputs, reg_conflicts] = find_fusion_regs(first, second);
    if (fusion_regs.empty() || reg_conflicts) {
        return {};
    }

    // Now that we know the fusion containers, resolve deferred locations
    if (state.producer_block_ == nullptr) {
        // Non-perfectly-nested producer (or perfectly-nested with multi-block body):
        // find write location for the first fusion container.
        // All fusion containers must be written at the same block for this to work.
        structured_control_flow::Block* common_block = nullptr;
        for (const auto& container : fusion_regs) {
            auto& fusion_arg = first.args.at(container);
            auto& nested_common = fusion_arg.nested_access;

            if (nested_common.wr_block.block_conflict) {
                return {};
            }
            if (!nested_common.wr_block.common_block) {
                return {};
            }
            if (!common_block) {
                common_block = nested_common.wr_block.common_block;
            } else if (common_block != nested_common.wr_block.common_block) {
                return {};
            }
        }
        if (common_block) {
            state.producer_block_ = common_block;
            state.producer_body_ = static_cast<Sequence*>(common_block->get_parent());
            state.producer_loops_ = collect_loop_parents(state.producer_body_, first.loop);
        } else {
            return {};
        }
    }

    if (!second_nested) {
        // Non-perfectly-nested consumer: find read location for the first fusion container
        // All fusion containers must be read at the same sequence for this to work
        structured_control_flow::Sequence* common_seq = nullptr;
        for (const auto& container : fusion_regs) {
            auto& fusion_arg = second.args.at(container);
            auto& nested_common = fusion_arg.nested_access;

            if (nested_common.rd_block.block_conflict) {
                return {};
            }
            if (!nested_common.rd_block.common_block) {
                return {};
            }
            if (!common_seq) {
                common_seq = static_cast<Sequence*>(nested_common.rd_block.common_block->get_parent());
            } else if (common_seq != nested_common.rd_block.common_block->get_parent()) {
                return {};
            }
        }
        if (common_seq) {
            state.consumer_body_ = common_seq;
            state.consumer_loops_ = collect_loop_parents(common_seq, second.loop);
        } else {
            return {};
        }
    }

    state.producer_fusion_candidate_ = get_fuse_candidate(*state.producer_loops_.back());
    state.consumer_fusion_candidate_ = get_fuse_candidate(*state.consumer_loops_.back());

    if (!state.consumer_fusion_candidate_ || !state.producer_fusion_candidate_) {
        DEBUG_PRINTLN(
            "Aborting fusion: Missing fusion candidate state for #" +
            std::to_string(state.producer_loops_.back()->element_id()) + " - #" +
            std::to_string(state.consumer_loops_.back()->element_id())
        );
        return {};
    }

    // Check if producer actually reads a fusion container in the dataflow.
    // If so, ProducerIntoConsumer is unsafe (original producer loop mutates the array
    // before the inlined copy reads it). Force ConsumerIntoProducer.
    if (state.direction_ == FusionDirection::ProducerIntoConsumer) {
        bool producer_reads_fusion = false;
        for (auto& container : fusion_regs) {
            auto& arg = state.producer_fusion_candidate_->args.at(container);
            if (arg.arg.is_explicit_input) {
                producer_reads_fusion = true;
                break;
            }
        }
        if (producer_reads_fusion) {
            state.direction_ = FusionDirection::ConsumerIntoProducer;
            // Re-check: consumer must be all-parallel for ConsumerIntoProducer
            if (!second_loop_info.is_perfectly_parallel) {
                return {};
            }
        }
    }

    // ProducerIntoConsumer only deep-copies producer_block_ into the consumer body.
    // If the producer body has multiple blocks (e.g. from prior BlockFusion merging
    // a previous fusion's writeback + inlined blocks), the write block may depend on
    // intermediates produced by earlier blocks that would NOT be copied. Reject.
    if (state.direction_ == FusionDirection::ProducerIntoConsumer && state.producer_body_->size() > 1) {
        return {};
    }

    if (state.direction_ == FusionDirection::ConsumerIntoProducer) {
        DEBUG_PRINTLN(
            "Aborting fusion, ConsumerIntoProducer still unsupported: #" +
            std::to_string(state.producer_body_->element_id()) + "<- #" +
            std::to_string(state.consumer_body_->element_id())
        );
        return {};
    }

    std::unordered_map<std::string, const data_flow::Subset*> producer_subsets;


    // For each fusion container, find the producer memlet and collect unique consumer subsets
    for (const auto& container : fusion_regs) {
        auto& arg = state.producer_fusion_candidate_->args.at(container);

        assert(arg.saw_access_locally()); // otherwise it should have never made it into this list to begin with

        auto common_subset = &arg.local_access.common_subset;
        if (arg.local_access.subsets_conflict || !common_subset) {
            DEBUG_PRINTLN(
                "Aborting fusion, conflicting write subsets on " + container + " in #" +
                std::to_string(state.producer_fusion_candidate_->loop->element_id())
            );
            return {};
        }
        // TODO old code aborted on finding access nodes of fusion regs that are read & write.
        //  cannot happen, because for now we do not allow any reads

        producer_subsets.emplace(container, &arg.local_access.common_subset.value());
    }

    transformations::FusionConsumerSubsetVisitor consumer_visitor(producer_subsets);
    bool abort = consumer_visitor.dispatch(*state.consumer_body_);
    if (abort) {
        return {};
    }

    // Get assumptions for the resolved write/read locations
    // Include trivial bounds from types to help delinearization with symbolic strides
    // TODO for the producer into consumer cases, it has to be innermost loops. Would not be for ConsumerIntoProducer.
    auto& producer_assumptions = state.producer_fusion_candidate_->assumptions;
    auto& consumer_assumptions = state.consumer_fusion_candidate_->assumptions;

    for (auto [container, unique_subsets] : consumer_visitor.unique_subsets_per_container()) {
        auto& producer_subset = *producer_subsets.at(container);
        // For each unique consumer subset, solve index mappings and create a FusionCandidate
        // The direction determines which side's indvars are solved for
        for (const auto& consumer_subset : unique_subsets) {
            std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> mappings;

            if (state.direction_ == FusionDirection::ProducerIntoConsumer) {
                // Solve producer indvars in terms of consumer indvars
                mappings = solve_subsets(
                    producer_subset,
                    consumer_subset,
                    state.producer_loops_,
                    state.consumer_loops_,
                    producer_assumptions,
                    consumer_assumptions
                );
            } else {
                // ConsumerIntoProducer: solve consumer indvars in terms of producer indvars
                // Arguments are swapped, so invert the range check direction
                mappings = solve_subsets(
                    consumer_subset,
                    producer_subset,
                    state.consumer_loops_,
                    state.producer_loops_,
                    consumer_assumptions,
                    producer_assumptions,
                    true
                );
            }

            if (mappings.empty()) {
                return {};
            }

            FusionRegCandidate candidate;
            candidate.container = container;
            candidate.consumer_subset = consumer_subset;
            candidate.index_mappings = std::move(mappings);

            state.fusion_candidates_.push_back(candidate);
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
        size_t first_sequential = state.consumer_loops_.size();
        for (size_t li = 0; li < state.consumer_loops_.size(); ++li) {
            if (dyn_cast<structured_control_flow::Map*>(state.consumer_loops_[li]) == nullptr) {
                sequential_indvars.insert(state.consumer_loops_[li]->indvar());
                if (first_sequential == state.consumer_loops_.size()) {
                    first_sequential = li;
                }
            }
        }
        if (sequential_indvars.empty()) {
            return {};
        }
        bool any_stream = false;
        bool any_init = false;
        for (const auto& candidate : state.fusion_candidates_) {
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
                    return {};
                }
                if (depends_on_sequential) {
                    return {};
                }
                any_init = true;
            } else {
                // Case 1 candidate: must be a streamed element.
                if (!depends_on_sequential) {
                    return {};
                }
                any_stream = true;
            }
        }
        // Do not mix patterns in a single fusion.
        if (any_init && any_stream) {
            return {};
        }
        if (any_init) {
            // Need an enclosing parallel band to host the hoisted init (the init must run
            // once per accumulator element, outside the sequential reduction loop).
            if (first_sequential == 0) {
                return {};
            }
            state.init_hoist_ = true;
            state.hoist_body_ = &state.consumer_loops_[first_sequential - 1]->root();
        }
    }

    state.domains_match = domains_match;
    if (!state.init_hoist_ && (!domains_match || state.direction_ != FusionDirection::ProducerIntoConsumer)) {
        // we will be copying a loop, so we can do our integrated RLE to simplify memory accesses
        for (auto& candidate : state.fusion_candidates_) {
            //
            candidate.integrated_rle = true;
        }
    }

    // Criterion: At least one valid fusion candidate
    if (!state.fusion_candidates_.empty()) {
        return std::move(state_ptr);
    } else {
        return {};
    }
}

ComplexFusionResult LoopFusionByAccessWorker::apply_fusion_by_access_plan(std::unique_ptr<Plan> plan_ptr) {
    auto& plan = *plan_ptr;

    if (plan.direction_ == FusionDirection::ProducerIntoConsumer) {
        return apply_producer_into_consumer(plan);
    } else {
        return apply_consumer_into_producer(plan);
    }
}

ComplexFusionResult LoopFusionByAccessWorker::apply_producer_into_consumer(Plan& plan) {
    auto& builder = this->builder();
    auto& sdfg = builder.subject();

    // Pattern 1 + Reverse Pattern 2: Inline producer blocks into consumer's read body
    auto& first_dataflow = plan.producer_block_->dataflow();

    // For each fusion candidate, create a temp and insert a producer block
    std::vector<std::string> candidate_temps;

    int rle_count = 0;
    for (size_t cand_idx = 0; cand_idx < plan.fusion_candidates_.size(); ++cand_idx) {
        auto& candidate = plan.fusion_candidates_[cand_idx];

        auto& container_type = sdfg.type(candidate.container);
        types::Scalar tmp_type(container_type.primitive_type());
        std::string temp_name;
        if (candidate.integrated_rle) {
            ++rle_count;
            // if we are forced to create a copy of the prod loop, then we can do integrated RLE
            // as it will reduce how many arguments we need to update
            // Case 1: scalarize the streamed element into a private temp.
            temp_name = builder.find_new_name("_fused_tmp");
            builder.add_container(temp_name, tmp_type);
            candidate_temps.push_back(temp_name);
        }

        // Insert the producer block at the beginning of the host sequence:
        //  - Case 1 (stream):     consumer_body_ = innermost sequential (reduction) loop body.
        //  - Case 2 (init-hoist): hoist_body_   = outer parallel-band body, before that loop.
        auto& host_seq = plan.consumer_target_sequence();
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
                if (access_node->data() == candidate.container && candidate.integrated_rle) {
                    // Case 1: redirect the producer's array write to the private scalar.
                    access_node->data(temp_name);
                } else if (access_node->data() == plan.first.indvar()->get_name()) {
                    // Determine the new expression for the index variable of the first map
                    symbolic::Expression new_expr = SymEngine::null;
                    for (auto& c : plan.fusion_candidates_) {
                        for (auto& [sym, expr] : c.index_mappings) {
                            if (symbolic::eq(sym, plan.first.indvar())) {
                                new_expr = expr;
                                break;
                            }
                        }
                        if (!new_expr.is_null()) {
                            break;
                        }
                    }

                    if (new_expr.is_null() || symbolic::eq(new_expr, plan.second.indvar())) {
                        // Simple case: The new expression is simply the index variable of the second loop
                        access_node->data(plan.second.indvar()->get_name());
                    } else {
                        // Complex case: Add AssignmentBlock before the new block (if necessary) and store the
                        // shifted index into a new temporary variable with an assignment. Then, replace the index
                        // variable with the new temporary variable
                        auto new_index_name = builder.find_new_name();
                        builder.add_container(new_index_name, builder.subject().type(plan.second.indvar()->get_name()));

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

            // Integrated RLE: the producer's array write becomes a scalar write (empty subset).
            // Default: keep the remapped array subset so the init writes the accumulator.
            auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
            if (dst_access != nullptr && dst_access->data() == candidate.container && candidate.integrated_rle &&
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

    DEBUG_PRINTLN(
        "Fusing loop stack by-access (#"
        << plan.first.element_id() << " | #" << plan.producer_loops_.back()->element_id() << " -> #"
        << plan.second.element_id() << " | #" << plan.consumer_loops_.back()->element_id()
        << (plan.init_hoist_ ? ", redu-init" : "") << (plan.domains_match ? ", domain-match" : "") << ", "
        << plan.fusion_candidates_.size() << " fRegs, " << rle_count << " RLEs"
        << ")"
    );

    // Integrated RLE: rewrite consumer reads of the fused arrays to the scalar temps.
    if (rle_count) {
        size_t num_producer_blocks = plan.fusion_candidates_.size();
        transformations::FusionConsumerUpdateVisitor update_visitor(builder, plan.fusion_candidates_, candidate_temps);
        update_visitor.dispatch_partial_sequence(*plan.consumer_body_, num_producer_blocks, plan.consumer_body_->size());
    }

    auto& ana = get_loop_analysis();
    auto& first_inner_info = ana.loop_info_local(plan.producer_loops_.back());

    ana.added_local_contents(
        plan.consumer_loops_.back(),
        first_inner_info.contains_side_effects,
        first_inner_info.contains_non_perfectly_nested
    );

    // innermost loops had to be leaf
    auto first_current = get_fuse_candidate(*plan.producer_loops_.back());
    auto second_current = get_fuse_candidate(*plan.consumer_target_loop());

    update_copied_leaf_contents_from_first_to_second(plan, first_current, second_current);

    bool removed_first = false;
    if ((!rle_count && plan.domains_match) || plan.init_hoist_) {
        ana.removed_loop(&plan.first);

        // We have moved all the relevant contents of producer loop, including any potential array writes, so
        builder.remove_from_parent(plan.first);
        removed_first = true;
    }

    return {
        .pattern_result =
            {.removed_first = removed_first, .visit_second_body = false, .second_root_replacement = nullptr},
        .fused = true
    };
}

ComplexFusionResult LoopFusionByAccessWorker::apply_consumer_into_producer(Plan& plan) {
    auto& builder = this->builder();
    auto& sdfg = builder.subject();

    throw std::runtime_error("ConsumerIntoProducer fusion not yet supported");

    DEBUG_PRINTLN(
        "Fusing " << plan.first.element_id() << " - " << plan.second.element_id() << " by "
                  << static_cast<int>(plan.direction_)
    );

    // ConsumerIntoProducer (Pattern 2): Inline consumer blocks into the producer's write body
    // Modify the producer block in-place to write to a temp scalar, add a writeback block
    // for the original array, then copy consumer blocks reading from the temp.

    std::vector<std::string> candidate_temps;
    auto& producer_dataflow = plan.producer_block_->dataflow();

    for (size_t cand_idx = 0; cand_idx < plan.fusion_candidates_.size(); ++cand_idx) {
        auto& candidate = plan.fusion_candidates_[cand_idx];

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
        auto& wb_block = builder.add_block_after(*plan.producer_body_, *plan.producer_block_);
        auto& wb_src = builder.add_access(wb_block, temp_name);
        auto& wb_dst = builder.add_access(wb_block, candidate.container);
        auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {});
        builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, original_write_subset);

        // Step 3: Copy consumer blocks after the writeback block
        structured_control_flow::ControlFlowNode* last_inserted = &wb_block;

        for (size_t i = 0; i < plan.consumer_body_->size(); ++i) {
            auto* consumer_block = dyn_cast<structured_control_flow::Block*>(&plan.consumer_body_->at(i));
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
            auto& new_block = builder.add_block_after(*plan.producer_body_, *last_inserted);
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
                    if (access_node->data() == plan.second.indvar()->get_name() &&
                        consumer_dataflow.in_degree(node) == 0) {
                        // Determine the new expression for the index variable of the second loop
                        symbolic::Expression new_expr = SymEngine::null;
                        for (auto& c : plan.fusion_candidates_) {
                            for (auto& [sym, expr] : c.index_mappings) {
                                if (symbolic::eq(sym, plan.second.indvar())) {
                                    new_expr = expr;
                                    break;
                                }
                            }
                            if (!new_expr.is_null()) {
                                break;
                            }
                        }

                        if (new_expr.is_null() || symbolic::eq(new_expr, plan.first.indvar())) {
                            // Simple case: The new expression is simply the index variable of the first map
                            access_node->data(plan.first.indvar()->get_name());
                        } else {
                            // Complex case: Add an AssignmentBlock (if necessary) and store the
                            // shifted index into a new temporary variable with an assignment. Then, replace the
                            // index variable with the new temporary variable
                            if (!init_assignment_block) {
                                init_assignment_block = &builder.add_assignments_at(*plan.producer_body_, 0, {});
                            }
                            auto new_index_name = builder.find_new_name();
                            builder
                                .add_container(new_index_name, builder.subject().type(plan.first.indvar()->get_name()));
                            init_assignment_block->assignments().insert({symbolic::symbol(new_index_name), new_expr});
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
    auto* parent = plan.second.get_parent();
    auto* parent_seq = dyn_cast<structured_control_flow::Sequence*>(parent);
    if (parent_seq != nullptr) {
        int idx = parent_seq->index(plan.second);
        if (idx >= 0) {
            builder.remove_child(*parent_seq, static_cast<size_t>(idx));
        }
    }
}

ComplexFusionResult LoopFusionByAccessWorker::
    try_fuse_by_access(FusionLoopCandidate& first, FusionLoopCandidate& second, bool domains_match) {
    auto plan = try_create_fusion_by_access_plan(first, second, domains_match);
    if (plan) {
        return apply_fusion_by_access_plan(std::move(plan));
    } else {
        return {};
    }
}

} // namespace sdfg::passes::loop_fusion
