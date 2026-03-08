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
    const symbolic::Assumptions& consumer_assumptions
) {
    // Delinearize subsets to recover multi-dimensional structure from linearized accesses
    // e.g. T[i*N + j] with assumptions on bounds -> T[i, j]
    auto producer_sub = symbolic::delinearize(producer_subset, producer_assumptions);
    auto consumer_sub = symbolic::delinearize(consumer_subset, consumer_assumptions);

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
    isl_space_free(unified);
    isl_ctx_free(ctx);

    if (!single_valued || !domain_covered) {
        return {};
    }

    return mappings;
}

bool MapFusion::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    fusion_candidates_.clear();

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
    // Criterion: First loop is perfectly nested
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto first_loop_info = loop_analysis.loop_info(&first_map_);
    if (!first_loop_info.is_perfectly_nested) {
        return false;
    }
    if (!first_loop_info.is_perfectly_parallel) {
        return false;
    }
    std::vector<structured_control_flow::StructuredLoop*> producer_loops = {&first_map_};
    structured_control_flow::Sequence* producer_body = &first_map_.root();
    structured_control_flow::ControlFlowNode* producer_node = &first_map_.root().at(0).first;
    while (auto* nested = dynamic_cast<structured_control_flow::Map*>(producer_node)) {
        producer_loops.push_back(nested);
        producer_body = &nested->root();
        producer_node = &nested->root().at(0).first;
    }

    // Criterion: Second loop is perfectly nested (but can have non-parallel loops)
    auto second_loop_info = loop_analysis.loop_info(&second_loop_);
    if (!second_loop_info.is_perfectly_nested) {
        return false;
    }
    std::vector<structured_control_flow::StructuredLoop*> consumer_loops = {&second_loop_};
    structured_control_flow::Sequence* consumer_body = &second_loop_.root();
    structured_control_flow::ControlFlowNode* consumer_node = &second_loop_.root().at(0).first;
    while (auto* nested = dynamic_cast<structured_control_flow::StructuredLoop*>(consumer_node)) {
        consumer_loops.push_back(nested);
        consumer_body = &nested->root();
        consumer_node = &nested->root().at(0).first;
    }

    // Get arguments analysis to identify inputs/outputs of each loop
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    auto first_args = arguments_analysis.arguments(analysis_manager, first_map_);
    auto second_args = arguments_analysis.arguments(analysis_manager, second_loop_);

    std::unordered_set<std::string> first_outputs;
    for (const auto& [name, arg] : first_args) {
        if (arg.is_output) {
            first_outputs.insert(name);
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
    }
    if (fusion_containers.empty()) {
        return false;
    }
    // Get assumptions for the innermost blocks (includes all enclosing loop bounds)
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& producer_assumptions = assumptions_analysis.get(*producer_node);
    auto& consumer_assumptions = assumptions_analysis.get(consumer_body->at(0).first);

    // For each fusion container, find the producer memlet and collect unique consumer subsets
    auto& first_dataflow = dynamic_cast<structured_control_flow::Block*>(producer_node)->dataflow();
    for (const auto& container : fusion_containers) {
        // Find unique producer in first map (producer)
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
        // Use a vector of subsets and deduplicate manually
        std::vector<data_flow::Subset> unique_subsets;
        for (size_t i = 0; i < consumer_body->size(); ++i) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&consumer_body->at(i).first);
            if (block == nullptr) {
                return false;
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
                        return false; // Dimension mismatch
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
        for (const auto& consumer_subset : unique_subsets) {
            auto mappings = solve_subsets(
                producer_subset,
                consumer_subset,
                producer_loops,
                consumer_loops,
                producer_assumptions,
                consumer_assumptions
            );
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

    // Navigate to the innermost block of the first map (handling nested maps)
    structured_control_flow::ControlFlowNode* first_block_node = &first_map_.root().at(0).first;
    while (auto* nested_map = dynamic_cast<structured_control_flow::Map*>(first_block_node)) {
        first_block_node = &nested_map->root().at(0).first;
    }
    auto* first_block = dynamic_cast<structured_control_flow::Block*>(first_block_node);
    auto& first_dataflow = first_block->dataflow();

    // Navigate to the innermost consumer sequence (handling nested loops)
    structured_control_flow::Sequence* second_root = &second_loop_.root();
    structured_control_flow::ControlFlowNode* consumer_node = &second_loop_.root().at(0).first;
    while (auto* nested = dynamic_cast<structured_control_flow::StructuredLoop*>(consumer_node)) {
        second_root = &nested->root();
        consumer_node = &nested->root().at(0).first;
    }

    // For each fusion candidate (unique container+subset pair), create a temp and insert a producer block
    // Track: candidate_index -> temp_name
    std::vector<std::string> candidate_temps;

    for (size_t cand_idx = 0; cand_idx < fusion_candidates_.size(); ++cand_idx) {
        auto& candidate = fusion_candidates_[cand_idx];

        // Create a temp scalar for this candidate
        auto& container_type = sdfg.type(candidate.container);
        std::string temp_name = builder.find_new_name("_fused_tmp");
        types::Scalar tmp_type(container_type.primitive_type());
        builder.add_container(temp_name, tmp_type);
        candidate_temps.push_back(temp_name);

        // Insert a producer block at the beginning of the consumer's innermost body
        auto& first_child = second_root->at(0).first;
        control_flow::Assignments empty_assignments;
        auto& producer_block = builder.add_block_before(*second_root, first_child, empty_assignments);

        // Deep copy all nodes from first block to producer block
        std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
        for (auto& node : first_dataflow.nodes()) {
            node_mapping[&node] = &builder.copy_node(producer_block, node);
            auto access = node_mapping[&node];
            if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(access)) {
                if (access_node->data() == candidate.container) {
                    // This is the producer access for this candidate - update to temp
                    access_node->data(temp_name);
                }
            }
        }

        // Add memlets with index substitution using this candidate's index_mappings
        for (auto& edge : first_dataflow.edges()) {
            auto& src_node = edge.src();
            auto& dst_node = edge.dst();

            // Substitute all producer indvars in subset
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

            // For output edges (to temp scalar), use empty subset
            auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
            if (dst_access != nullptr && dst_access->data() == candidate.container &&
                first_dataflow.in_degree(*dst_access) > 0) {
                new_subset.clear(); // Scalar has empty subset
                base_type = &tmp_type;
            }

            builder.add_memlet(
                producer_block,
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

    // Now update all read accesses in consumer blocks to point to the appropriate temp
    // We need to match each access node's memlet subset to find the right candidate
    size_t num_producer_blocks = fusion_candidates_.size();

    for (size_t block_idx = num_producer_blocks; block_idx < second_root->size(); ++block_idx) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&second_root->at(block_idx).first);
        if (block == nullptr) {
            continue;
        }

        auto& dataflow = block->dataflow();

        // Find all read access nodes
        for (auto& node : dataflow.nodes()) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
            if (access == nullptr) {
                continue;
            }

            // Only update read accesses (out_degree > 0)
            if (dataflow.out_degree(*access) == 0) {
                continue;
            }

            // Capture original container name before any modifications
            std::string original_container = access->data();

            // Check each outgoing memlet to find which candidates match
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() != data_flow::MemletType::Computational) {
                    continue;
                }

                const auto& memlet_subset = memlet.subset();

                // Find matching candidate by container and subset
                for (size_t cand_idx = 0; cand_idx < fusion_candidates_.size(); ++cand_idx) {
                    auto& candidate = fusion_candidates_[cand_idx];

                    if (original_container != candidate.container) {
                        continue;
                    }

                    // Check if subset matches
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

                    // Found a match - update the access node and memlet
                    const auto& temp_name = candidate_temps[cand_idx];
                    auto& temp_type = sdfg.type(temp_name);

                    access->data(temp_name);

                    // Update this memlet
                    memlet.set_subset({});
                    memlet.set_base_type(temp_type);

                    // Also update any incoming edges
                    for (auto& in_edge : dataflow.in_edges(*access)) {
                        in_edge.set_subset({});
                        in_edge.set_base_type(temp_type);
                    }

                    break; // Found the matching candidate
                }
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
