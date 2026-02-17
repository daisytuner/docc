#include "sdfg/transformations/map_fusion.h"

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace transformations {

MapFusion::MapFusion(structured_control_flow::Map& first_map, structured_control_flow::StructuredLoop& second_loop)
    : first_map_(first_map), second_loop_(second_loop) {}

std::string MapFusion::name() const { return "MapFusion"; }

bool MapFusion::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    fusion_candidates_.clear();

    // Criterion: Get parent scope and verify both loops are sequential children
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* first_parent = scope_analysis.parent_scope(&first_map_);
    auto* second_parent = scope_analysis.parent_scope(&second_loop_);

    if (first_parent == nullptr || second_parent == nullptr) {
        return false;
    }

    // Both must have the same parent sequence
    if (first_parent != second_parent) {
        return false;
    }

    auto* parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(first_parent);
    if (parent_sequence == nullptr) {
        return false;
    }

    // Criterion: first_map must immediately precede second_loop in the sequence
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

    // Criterion: First map must have simple body (single block)
    // This ensures no WAR/WAW hazards when replicating the producer
    if (first_map_.root().size() != 1) {
        return false;
    }

    auto* first_block = dynamic_cast<structured_control_flow::Block*>(&first_map_.root().at(0).first);
    if (first_block == nullptr) {
        return false;
    }

    // Criterion: First block's transition should be empty
    if (!first_map_.root().at(0).second.empty()) {
        return false;
    }

    // Get arguments analysis to identify inputs/outputs of each loop
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    auto first_args = arguments_analysis.arguments(analysis_manager, first_map_);
    auto second_args = arguments_analysis.arguments(analysis_manager, second_loop_);

    // Find containers that are outputs of first map and inputs of second map
    std::unordered_set<std::string> first_outputs;
    for (const auto& [name, arg] : first_args) {
        if (arg.is_output) {
            first_outputs.insert(name);
        }
    }

    std::unordered_set<std::string> fusion_containers;
    for (const auto& [name, arg] : second_args) {
        if (arg.is_input && first_outputs.contains(name)) {
            // Criterion: Skip scalars - only fuse array/pointer accesses
            if (arg.is_scalar) {
                continue;
            }
            fusion_containers.insert(name);
        }
    }
    if (fusion_containers.empty()) {
        return false;
    }

    // Analyze memory access patterns for each fusion candidate
    auto& first_dataflow = first_block->dataflow();

    auto first_indvar = first_map_.indvar();
    auto second_indvar = second_loop_.indvar();

    // For each fusion container, find the producer memlet and collect unique consumer subsets
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
        // Assume linearized subset
        if (producer_subset.size() != 1) {
            return false;
        }

        // Extract affine coefficients for producer (done once per container)
        auto producer_expr = producer_subset.at(0);
        symbolic::SymbolVec producer_symbols = {first_indvar};
        auto producer_poly = symbolic::polynomial(producer_expr, producer_symbols);
        if (producer_poly.is_null()) {
            return false; // Not a polynomial
        }
        auto producer_coeffs = symbolic::affine_coefficients(producer_poly, producer_symbols);
        if (producer_coeffs.empty()) {
            return false; // Not affine
        }
        auto first_coeff = producer_coeffs[first_indvar];
        if (symbolic::eq(first_coeff, symbolic::zero())) {
            return false;
        }
        auto producer_constant = producer_coeffs[symbolic::symbol("__daisy_constant__")];

        // Collect all unique (container, subset) pairs from consumer blocks
        // A subset is considered unique if it's not symbolically equal to any existing one
        symbolic::ExpressionSet unique_subsets;

        for (size_t i = 0; i < second_loop_.root().size(); ++i) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&second_loop_.root().at(i).first);
            if (block == nullptr) {
                return false;
            }

            auto& dataflow = block->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }

                // Check all read memlets from this access
                for (auto& memlet : dataflow.out_edges(*access)) {
                    if (memlet.type() != data_flow::MemletType::Computational) {
                        return false;
                    }

                    auto& consumer_subset = memlet.subset();
                    // Assume linearized subset
                    if (consumer_subset.size() != 1) {
                        return false;
                    }
                    unique_subsets.insert(consumer_subset.at(0));
                }
            }
        }

        // For each unique subset, compute index_mapping and create a FusionCandidate
        for (const auto& consumer_expr : unique_subsets) {
            // index_mapping = (consumer_expr - producer_constant) / first_coeff
            auto numerator = symbolic::sub(consumer_expr, producer_constant);
            auto index_mapping = symbolic::div(numerator, first_coeff);
            index_mapping = symbolic::expand(index_mapping);

            // Verify the mapping is valid (contains only second_indvar and constants)
            bool valid_mapping = true;
            for (const auto& atom : symbolic::atoms(index_mapping)) {
                std::string name = atom->get_name();
                if (name != second_indvar->get_name()) {
                    // Check if it's a parameter (constant for both loops)
                    bool is_param = false;
                    for (const auto& [arg_name, arg] : second_args) {
                        if (arg_name == name && arg.is_scalar && !arg.is_output) {
                            is_param = true;
                            break;
                        }
                    }
                    if (!is_param) {
                        for (const auto& [arg_name, arg] : first_args) {
                            if (arg_name == name && arg.is_scalar && !arg.is_output) {
                                is_param = true;
                                break;
                            }
                        }
                    }
                    if (!is_param) {
                        valid_mapping = false;
                        break;
                    }
                }
            }

            if (!valid_mapping) {
                return false;
            }

            // Store the fusion candidate
            FusionCandidate candidate;
            candidate.container = container;
            candidate.consumer_subset = {consumer_expr};
            candidate.index_mapping = index_mapping;

            fusion_candidates_.push_back(candidate);
        }
    }

    // Criterion: At least one valid fusion candidate
    return !fusion_candidates_.empty();
}

void MapFusion::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Get the producer block from first map
    auto* first_block = dynamic_cast<structured_control_flow::Block*>(&first_map_.root().at(0).first);
    auto& first_dataflow = first_block->dataflow();

    auto first_indvar = first_map_.indvar();

    auto& second_root = second_loop_.root();

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

        // Insert a producer block at the beginning of the second loop's body
        auto& first_child = second_root.at(0).first;
        control_flow::Assignments empty_assignments;
        auto& producer_block = builder.add_block_before(second_root, first_child, empty_assignments);

        // Deep copy all nodes from first block to producer block
        std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;

        for (auto& node : first_dataflow.nodes()) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
            if (access != nullptr) {
                // Check if this is an output access node for this candidate's container
                if (first_dataflow.in_degree(*access) > 0 && access->data() == candidate.container) {
                    // Replace with temp scalar access
                    auto& temp_access = builder.add_access(producer_block, temp_name);
                    node_mapping[&node] = &temp_access;
                } else {
                    // Copy the access node as-is
                    node_mapping[&node] = &builder.copy_node(producer_block, node);
                }
            } else {
                // Copy other nodes (tasklets, library nodes, etc.)
                node_mapping[&node] = &builder.copy_node(producer_block, node);
            }
        }

        // Add memlets with index substitution using this candidate's index_mapping
        for (auto& edge : first_dataflow.edges()) {
            auto& src_node = edge.src();
            auto& dst_node = edge.dst();

            // Substitute indices in subset
            data_flow::Subset new_subset;
            for (const auto& dim : edge.subset()) {
                auto new_dim = symbolic::subs(dim, first_indvar, candidate.index_mapping);
                new_dim = symbolic::expand(new_dim);
                new_subset.push_back(new_dim);
            }

            // For output edges (to temp scalar), use empty subset
            auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
            if (dst_access != nullptr && dst_access->data() == candidate.container &&
                first_dataflow.in_degree(*dst_access) > 0) {
                new_subset.clear(); // Scalar has empty subset
            }

            builder.add_memlet(
                producer_block,
                *node_mapping[&src_node],
                edge.src_conn(),
                *node_mapping[&dst_node],
                edge.dst_conn(),
                new_subset,
                edge.base_type(),
                edge.debug_info()
            );
        }
    }

    // Now update all read accesses in consumer blocks to point to the appropriate temp
    // We need to match each access node's memlet subset to find the right candidate
    size_t num_producer_blocks = fusion_candidates_.size();

    for (size_t block_idx = num_producer_blocks; block_idx < second_root.size(); ++block_idx) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&second_root.at(block_idx).first);
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
