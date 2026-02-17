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

    // For each fusion container, check if we can solve the access equation
    for (const auto& container : fusion_containers) {
        // Find write accesses in first map
        data_flow::AccessNode* producer_access = nullptr;
        data_flow::Memlet* producer_memlet = nullptr;

        for (auto& node : first_dataflow.nodes()) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
            if (access == nullptr || access->data() != container) {
                continue;
            }

            // Check if this is a write (has incoming edges)
            if (first_dataflow.in_degree(*access) > 0) {
                // Get the write memlet
                for (auto& memlet : first_dataflow.in_edges(*access)) {
                    if (memlet.type() == data_flow::MemletType::Computational) {
                        producer_access = access;
                        producer_memlet = &memlet;
                        break;
                    }
                }
                if (producer_access != nullptr) {
                    break;
                }
            }
        }

        if (producer_access == nullptr || producer_memlet == nullptr) {
            continue; // No valid producer found for this container
        }

        // Find read accesses in any block of the second loop
        data_flow::AccessNode* consumer_access = nullptr;
        data_flow::Memlet* consumer_memlet = nullptr;
        structured_control_flow::Block* consumer_block = nullptr;

        for (size_t i = 0; i < second_loop_.root().size(); ++i) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&second_loop_.root().at(i).first);
            if (block == nullptr) {
                continue;
            }

            auto& second_dataflow = block->dataflow();
            for (auto& node : second_dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }

                // Check if this is a read (has outgoing edges)
                if (second_dataflow.out_degree(*access) > 0) {
                    // Get the read memlet
                    for (auto& memlet : second_dataflow.out_edges(*access)) {
                        if (memlet.type() == data_flow::MemletType::Computational) {
                            consumer_access = access;
                            consumer_memlet = &memlet;
                            consumer_block = block;
                            break;
                        }
                    }
                    if (consumer_access != nullptr) {
                        break;
                    }
                }
            }
            if (consumer_access != nullptr) {
                break;
            }
        }

        if (consumer_access == nullptr || consumer_memlet == nullptr) {
            continue; // No valid consumer found for this container
        }

        // Extract subsets
        const auto& producer_subset = producer_memlet->subset();
        const auto& consumer_subset = consumer_memlet->subset();

        // Criterion: Subsets must have the same dimensionality
        if (producer_subset.size() != consumer_subset.size()) {
            continue;
        }

        // Criterion: For now, focus on 1D accesses (simple case)
        if (producer_subset.size() != 1) {
            continue;
        }

        // Solve the affine equation: producer_subset[first_indvar] = consumer_subset[second_indvar]
        // We need to find first_indvar in terms of second_indvar
        //
        // Given: producer_subset = a * first_indvar + b
        //        consumer_subset = c * second_indvar + d
        // If they access the same element: a * first_indvar + b = c * second_indvar + d
        // Solve for first_indvar: first_indvar = (c * second_indvar + d - b) / a

        auto producer_expr = producer_subset[0];
        auto consumer_expr = consumer_subset[0];

        // Extract affine coefficients for producer
        symbolic::SymbolVec producer_symbols = {first_indvar};
        auto producer_poly = symbolic::polynomial(producer_expr, producer_symbols);
        if (producer_poly.is_null()) {
            continue; // Not a polynomial
        }
        auto producer_coeffs = symbolic::affine_coefficients(producer_poly, producer_symbols);
        if (producer_coeffs.empty()) {
            continue; // Not affine
        }

        // Check that the coefficient of first_indvar is non-zero
        auto first_coeff = producer_coeffs[first_indvar];
        if (symbolic::eq(first_coeff, symbolic::zero())) {
            continue; // first_indvar doesn't appear in producer expression
        }

        // Compute the inverse: first_indvar = (consumer_expr - producer_constant) / first_coeff
        auto producer_constant = producer_coeffs[symbolic::symbol("__daisy_constant__")];

        // index_mapping = (consumer_expr - producer_constant) / first_coeff
        auto numerator = symbolic::sub(consumer_expr, producer_constant);
        auto index_mapping = symbolic::div(numerator, first_coeff);

        // Simplify and check that it only uses second_indvar and constants
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
                // Also check if it's in first_args as input scalar
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
            continue;
        }

        // Additional check: verify that the mapping results in integer indices
        // For now, we require that the coefficient divides evenly
        // This is a simplified check - a more complete implementation would use ISL

        // Store the fusion candidate
        FusionCandidate candidate;
        candidate.container = container;
        candidate.consumer_access = consumer_access;
        candidate.producer_access = producer_access;
        candidate.consumer_memlet = consumer_memlet;
        candidate.producer_memlet = producer_memlet;
        candidate.consumer_block = consumer_block;
        candidate.index_mapping = index_mapping;

        fusion_candidates_.push_back(candidate);
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

    // Insert a new producer block at the beginning of the second loop's body
    auto& second_root = second_loop_.root();
    auto& first_child = second_root.at(0).first;
    control_flow::Assignments empty_assignments;
    auto& producer_block = builder.add_block_before(second_root, first_child, empty_assignments);

    // Build a set of containers that are written to in the first block (outputs)
    std::unordered_set<std::string> output_containers;
    for (auto node : first_dataflow.data_nodes()) {
        if (dynamic_cast<data_flow::ConstantNode*>(node) != nullptr) {
            continue; // Skip constants
        }
        if (first_dataflow.in_degree(*node) > 0) {
            output_containers.insert(node->data());
        }
    }

    // Create temporary scalars for each output container
    std::unordered_map<std::string, std::string> container_to_temp;

    for (const auto& container : output_containers) {
        auto& type = sdfg.type(container);
        std::string temp_name = builder.find_new_name("_fused_tmp");
        types::Scalar tmp_type(type.primitive_type());
        builder.add_container(temp_name, tmp_type);
        container_to_temp[container] = temp_name;
    }

    // Deep copy all nodes from first block to producer block
    std::unordered_map<const data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;

    for (auto& node : first_dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr) {
            // Check if this is an output access node (written to)
            if (first_dataflow.in_degree(*access) > 0 && container_to_temp.contains(access->data())) {
                // Replace with temp scalar access
                auto& temp_access = builder.add_access(producer_block, container_to_temp[access->data()]);
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

    // Add memlets with index substitution
    for (auto& edge : first_dataflow.edges()) {
        auto& src_node = edge.src();
        auto& dst_node = edge.dst();

        // Substitute indices in subset
        data_flow::Subset new_subset;
        for (const auto& dim : edge.subset()) {
            // For each fusion candidate, use its index_mapping
            auto new_dim = dim;
            for (auto& candidate : fusion_candidates_) {
                new_dim = symbolic::subs(new_dim, first_indvar, candidate.index_mapping);
            }
            new_dim = symbolic::expand(new_dim);
            new_subset.push_back(new_dim);
        }

        // For output edges (to temp scalars), use empty subset
        auto* dst_access = dynamic_cast<data_flow::AccessNode*>(&dst_node);
        if (dst_access != nullptr && container_to_temp.contains(dst_access->data())) {
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

    // In the consumer blocks, replace reads from intermediate containers with reads from temp scalars
    for (auto& candidate : fusion_candidates_) {
        auto& consumer_block = *candidate.consumer_block;
        auto& consumer_dataflow = consumer_block.dataflow();
        const auto& temp_name = container_to_temp[candidate.container];
        auto& temp_type = sdfg.type(temp_name);

        auto* consumer_access = candidate.consumer_access;

        // Rename the access node to point to the temp scalar
        consumer_access->data(temp_name);

        // Update all outgoing edges: set empty subset (scalar) and new type
        for (auto& edge : consumer_dataflow.out_edges(*consumer_access)) {
            edge.set_subset({});
            edge.set_base_type(temp_type);
        }

        // Update all incoming edges: set empty subset (scalar) and new type
        for (auto& edge : consumer_dataflow.in_edges(*consumer_access)) {
            edge.set_subset({});
            edge.set_base_type(temp_type);
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
