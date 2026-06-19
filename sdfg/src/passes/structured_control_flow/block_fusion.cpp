#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include <cstddef>
#include <unordered_set>
#include <utility>
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"

namespace sdfg {
namespace passes {

BlockFusion::BlockFusion(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool ignore_libnodes
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), ignore_libnodes_(ignore_libnodes) {}

bool BlockFusion::can_be_applied(
    data_flow::DataFlowGraph& first_graph,
    control_flow::Assignments& first_assignments,
    data_flow::DataFlowGraph& second_graph,
    control_flow::Assignments& second_assignments
) {
    // Criterion: No side-effect nodes
    std::unordered_set<std::string> first_write_symbols;
    for (auto& node : first_graph.nodes()) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            if (this->ignore_libnodes_ || lib_node->side_effect()) {
                return false;
            }
        } else if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (first_graph.in_degree(*access_node) > 0) {
                auto& type = builder_.subject().type(access_node->data());
                if (type.is_symbol()) {
                    first_write_symbols.insert(access_node->data());
                }
            }
        }
    }
    std::unordered_set<std::string> second_write_symbols;
    for (auto& node : second_graph.nodes()) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            if (this->ignore_libnodes_ || lib_node->side_effect()) {
                return false;
            }
        } else if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (second_graph.in_degree(*access_node) > 0) {
                auto& type = builder_.subject().type(access_node->data());
                if (type.is_symbol()) {
                    second_write_symbols.insert(access_node->data());
                }
            }
        }
    }

    // Criterion: Subsets and library node symbols may not use written symbols
    for (auto& edge : first_graph.edges()) {
        for (auto& dim : edge.subset()) {
            for (auto& sym : symbolic::atoms(dim)) {
                if (second_write_symbols.find(sym->get_name()) != second_write_symbols.end()) {
                    return false;
                }
            }
        }
    }
    for (auto* libnode : first_graph.library_nodes()) {
        for (auto& sym : libnode->symbols()) {
            if (second_write_symbols.find(sym->get_name()) != second_write_symbols.end()) {
                return false;
            }
        }
    }
    for (auto& edge : second_graph.edges()) {
        for (auto& dim : edge.subset()) {
            for (auto& sym : symbolic::atoms(dim)) {
                if (first_write_symbols.find(sym->get_name()) != first_write_symbols.end()) {
                    return false;
                }
            }
        }
    }
    for (auto* libnode : second_graph.library_nodes()) {
        for (auto& sym : libnode->symbols()) {
            if (first_write_symbols.find(sym->get_name()) != first_write_symbols.end()) {
                return false;
            }
        }
    }

    // Criterion: Transition must be empty
    if (!first_assignments.empty()) {
        return false;
    }

    // Criterion: Keep references and dereference in separate blocks
    for (auto& edge : first_graph.edges()) {
        if (edge.type() != data_flow::MemletType::Computational) {
            return false;
        }
    }
    for (auto& edge : second_graph.edges()) {
        if (edge.type() != data_flow::MemletType::Computational) {
            return false;
        }
    }

    // Determine sets of weakly connected components for first graph
    auto [first_num_components, first_components] = first_graph.weakly_connected_components();
    std::vector<std::unordered_set<const data_flow::AccessNode*>> first_weakly_connected(first_num_components);
    for (auto comp : first_components) {
        // Only handle access nodes of the first graph
        if (dynamic_cast<const data_flow::ConstantNode*>(comp.first)) {
            continue;
        } else if (auto* access_node = dynamic_cast<const data_flow::AccessNode*>(comp.first)) {
            first_weakly_connected[comp.second].insert(access_node);
        }
    }

    // Determine sets of weakly connected components for second graph
    auto [second_num_components, second_components] = second_graph.weakly_connected_components();
    std::vector<std::unordered_set<const data_flow::AccessNode*>> second_weakly_connected(second_num_components);
    for (auto comp : second_components) {
        // Only handle access nodes of the second graph
        if (dynamic_cast<const data_flow::ConstantNode*>(comp.first)) {
            continue;
        } else if (auto* access_node = dynamic_cast<const data_flow::AccessNode*>(comp.first)) {
            second_weakly_connected[comp.second].insert(access_node);
        }
    }

    // For each combination of weakly connected components:
    for (size_t first = 0; first < first_num_components; first++) {
        for (size_t second = 0; second < second_num_components; second++) {
            // Match all access nodes with the same container
            std::vector<std::pair<const data_flow::AccessNode*, const data_flow::AccessNode*>> matches;
            for (auto* first_access_node : first_weakly_connected[first]) {
                for (auto* second_access_node : second_weakly_connected[second]) {
                    if (first_access_node->data() == second_access_node->data()) {
                        matches.push_back({first_access_node, second_access_node});
                    }
                }
            }
            // Skip if there are no matches
            if (matches.empty()) {
                continue;
            }
            // There must be at least one sink in the first graph and one source in the second graph that match
            bool connection = false;
            for (auto [first_access_node, second_access_node] : matches) {
                if (first_graph.out_degree(*first_access_node) == 0 &&
                    second_graph.in_degree(*second_access_node) == 0) {
                    connection = true;
                    break;
                }
            }
            if (!connection) {
                return false;
            }
        }
    }

    return true;
};

void BlockFusion::apply(
    structured_control_flow::Block& first_block,
    control_flow::Assignments& first_assignments,
    structured_control_flow::Block& second_block,
    control_flow::Assignments& second_assignments
) {
    data_flow::DataFlowGraph& first_graph = first_block.dataflow();
    data_flow::DataFlowGraph& second_graph = second_block.dataflow();

    // Update symbols
    for (auto& entry : second_assignments) {
        first_assignments[entry.first] = entry.second;
    }

    // Collect nodes to connect to
    auto pdoms = first_graph.post_dominators();
    std::unordered_map<data_flow::AccessNode*, data_flow::AccessNode*> connectors;
    for (auto& node : second_graph.sources()) {
        if (!dynamic_cast<data_flow::AccessNode*>(node)) {
            continue;
        }
        auto access_node = static_cast<data_flow::AccessNode*>(node);

        // Not used in first graph
        if (!pdoms.contains(access_node->data())) {
            continue;
        }

        connectors[access_node] = pdoms.at(access_node->data());
    }

    // Copy nodes from second to first
    std::unordered_map<data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
    for (auto& node : second_graph.nodes()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (connectors.contains(access_node)) {
                // Connect by replacement
                node_mapping[access_node] = connectors[access_node];
            } else {
                node_mapping[access_node] = &builder_.copy_node(first_block, *access_node);
            }
        } else {
            node_mapping[&node] = &builder_.copy_node(first_block, node);
        }
    }

    // Connect new nodes according to edges of second graph
    for (auto& edge : second_graph.edges()) {
        auto& src_node = edge.src();
        auto& dst_node = edge.dst();

        builder_.add_memlet(
            first_block,
            *node_mapping[&src_node],
            edge.src_conn(),
            *node_mapping[&dst_node],
            edge.dst_conn(),
            edge.subset(),
            edge.base_type(),
            edge.debug_info()
        );
    }

    builder_.merge_sinks(first_block);
};

bool BlockFusion::accept(structured_control_flow::Sequence& node) {
    bool applied = false;

    if (node.size() == 0) {
        return applied;
    }

    // Traverse node to find pairs of blocks
    size_t i = 0;
    while (i < (node.size() - 1)) {
        auto current_entry = node.at(i);
        if (dynamic_cast<structured_control_flow::Block*>(&current_entry.first) == nullptr) {
            i++;
            continue;
        }
        auto current_block = static_cast<structured_control_flow::Block*>(&current_entry.first);

        auto next_entry = node.at(i + 1);
        if (dynamic_cast<structured_control_flow::Block*>(&next_entry.first) == nullptr) {
            i++;
            continue;
        }
        auto next_block = static_cast<structured_control_flow::Block*>(&next_entry.first);

        if (this->can_be_applied(
                current_block->dataflow(),
                current_entry.second.assignments(),
                next_block->dataflow(),
                next_entry.second.assignments()
            )) {
            this->apply(*current_block, current_entry.second.assignments(), *next_block, next_entry.second.assignments());
            builder_.remove_child(node, i + 1);
            applied = true;

            continue;
        }

        i++;
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
