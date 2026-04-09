#include "sdfg/transformations/map_fusion.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>
#include <symengine/solve.h>
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace transformations {

MapFusion::MapFusion(structured_control_flow::Map& first_map, structured_control_flow::StructuredLoop& second_loop)
    : first_map_(first_map), second_loop_(second_loop) {}

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
        auto& child = seq.at(i).first;

        if (auto* blk = dynamic_cast<structured_control_flow::Block*>(&child)) {
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
        } else if (auto* nested_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
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
        auto& child = seq.at(i).first;

        if (auto* blk = dynamic_cast<structured_control_flow::Block*>(&child)) {
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
        } else if (auto* nested_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
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
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* first_parent = scope_analysis.parent_scope(&first_map_);
    auto* second_parent = scope_analysis.parent_scope(&second_loop_);
    if (first_parent == nullptr || second_parent == nullptr) {
        return false;
    }
    if (first_parent != second_parent) {
        return false;
    }

    auto* parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(first_parent);
    if (parent_sequence == nullptr) {
        return false;
    }

    int first_index = parent_sequence->index(first_map_);
    int second_index = parent_sequence->index(second_loop_);
    if (first_index == -1 || second_index == -1) {
        return false;
    }
    if (second_index != first_index + 1) {
        return false;
    }

    // Criterion: Transition between maps should have no assignments
    auto& transition = parent_sequence->at(first_index).second;
    if (!transition.empty()) {
        return false;
    }
    // Determine fusion pattern based on nesting properties
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto first_loop_info = loop_analysis.loop_info(&first_map_);
    auto second_loop_info = loop_analysis.loop_info(&second_loop_);

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
    // ProducerIntoConsumer: producer is replicated at each consumer site — producer must be all-parallel.
    // ConsumerIntoProducer: consumer is reordered into producer's nest — consumer must be all-parallel.
    if (direction_ == FusionDirection::ProducerIntoConsumer) {
        if (!first_loop_info.is_perfectly_parallel) {
            return false;
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
        structured_control_flow::ControlFlowNode* node = &first_map_.root().at(0).first;
        while (auto* nested = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
            producer_loops_.push_back(nested);
            producer_body_ = &nested->root();
            node = &nested->root().at(0).first;
        }
        producer_block_ = dynamic_cast<structured_control_flow::Block*>(node);
        if (producer_block_ == nullptr) {
            return false;
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
        // Perfectly nested: walk the at(0).first chain
        consumer_loops_.push_back(&second_loop_);
        consumer_body_ = &second_loop_.root();
        structured_control_flow::ControlFlowNode* node = &second_loop_.root().at(0).first;
        while (auto* nested = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
            consumer_loops_.push_back(nested);
            consumer_body_ = &nested->root();
            node = &nested->root().at(0).first;
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

    std::unordered_set<std::string> fusion_containers;
    for (const auto& [name, arg] : second_args) {
        if (first_outputs.contains(name)) {
            if (arg.is_output) {
                return false;
            }
            if (arg.is_input) {
                fusion_containers.insert(name);
            }
        }
        if (first_inputs.contains(name) && arg.is_output) {
            return false;
        }
    }
    if (fusion_containers.empty()) {
        return false;
    }

    // Now that we know the fusion containers, resolve deferred locations
    if (!first_nested) {
        // Non-perfectly-nested producer: find write location for the first fusion container
        // All fusion containers must be written at the same block for this to work
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
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& producer_assumptions = assumptions_analysis.get(*producer_block_);
    auto& consumer_assumptions = assumptions_analysis.get(consumer_body_->at(0).first);

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
        }

        // Collect all unique subsets from consumer blocks
        std::vector<data_flow::Subset> unique_subsets;
        for (size_t i = 0; i < consumer_body_->size(); ++i) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&consumer_body_->at(i).first);
            if (block == nullptr) {
                // Skip non-block children (e.g. nested loops that are not related)
                continue;
            }

            auto& dataflow = block->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }
                if (dataflow.in_degree(*access) != 0 || dataflow.out_degree(*access) == 0) {
                    return false;
                }

                // Check all read memlets from this access
                for (auto& memlet : dataflow.out_edges(*access)) {
                    if (memlet.type() != data_flow::MemletType::Computational) {
                        return false;
                    }

                    auto& consumer_subset = memlet.subset();
                    if (consumer_subset.size() != producer_subset.size()) {
                        return false;
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
        }

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
            std::string temp_name = builder.find_new_name("_fused_tmp");
            types::Scalar tmp_type(container_type.primitive_type());
            builder.add_container(temp_name, tmp_type);
            candidate_temps.push_back(temp_name);

            // Insert a producer block at the beginning of the consumer's body
            auto& first_child = consumer_body_->at(0).first;
            control_flow::Assignments empty_assignments;
            auto& new_block = builder.add_block_before(*consumer_body_, first_child, empty_assignments);

            // Deep copy all nodes from producer block to new block
            std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
            for (auto& node : first_dataflow.nodes()) {
                node_mapping[&node] = &builder.copy_node(new_block, node);
                auto* copied = node_mapping[&node];
                if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(copied)) {
                    if (access_node->data() == candidate.container) {
                        access_node->data(temp_name);
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

                // For output edges to temp scalar, use empty subset
                auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
                if (dst_access != nullptr && dst_access->data() == candidate.container &&
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

        // Update all read accesses in consumer blocks to point to the appropriate temp
        size_t num_producer_blocks = fusion_candidates_.size();

        for (size_t block_idx = num_producer_blocks; block_idx < consumer_body_->size(); ++block_idx) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&consumer_body_->at(block_idx).first);
            if (block == nullptr) {
                continue;
            }

            auto& dataflow = block->dataflow();

            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || dataflow.out_degree(*access) == 0) {
                    continue;
                }

                std::string original_container = access->data();

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

                        if (!subset_matches) {
                            continue;
                        }

                        const auto& temp_name = candidate_temps[cand_idx];
                        auto& temp_type = sdfg.type(temp_name);

                        access->data(temp_name);

                        memlet.set_subset({});
                        memlet.set_base_type(temp_type);

                        for (auto& in_edge : dataflow.in_edges(*access)) {
                            in_edge.set_subset({});
                            in_edge.set_base_type(temp_type);
                        }

                        break;
                    }
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
            control_flow::Assignments empty_assignments;
            auto& wb_block = builder.add_block_after(*producer_body_, *producer_block_, empty_assignments);
            auto& wb_src = builder.add_access(wb_block, temp_name);
            auto& wb_dst = builder.add_access(wb_block, candidate.container);
            auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {});
            builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, original_write_subset);

            // Step 3: Copy consumer blocks after the writeback block
            structured_control_flow::ControlFlowNode* last_inserted = &wb_block;

            for (size_t i = 0; i < consumer_body_->size(); ++i) {
                auto* consumer_block = dynamic_cast<structured_control_flow::Block*>(&consumer_body_->at(i).first);
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
                auto& new_block = builder.add_block_after(*producer_body_, *last_inserted, empty_assignments);

                // Deep copy all nodes from consumer block
                std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
                for (auto& node : consumer_dataflow.nodes()) {
                    node_mapping[&node] = &builder.copy_node(new_block, node);
                    auto* copied = node_mapping[&node];
                    if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(copied)) {
                        if (access_node->data() == candidate.container) {
                            access_node->data(temp_name);
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
        auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
        auto* parent = scope_analysis.parent_scope(&second_loop_);
        auto* parent_seq = dynamic_cast<structured_control_flow::Sequence*>(parent);
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
    std::string second_type = "for";
    if (dynamic_cast<structured_control_flow::Map*>(&second_loop_) != nullptr) {
        second_type = "map";
    }
    j["transformation_type"] = this->name();
    j["subgraph"] = {
        {"0", {{"element_id", first_map_.element_id()}, {"type", "map"}}},
        {"1", {{"element_id", second_loop_.element_id()}, {"type", second_type}}}
    };
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

    auto* first_map = dynamic_cast<structured_control_flow::Map*>(first_element);
    auto* second_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(second_element);

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
