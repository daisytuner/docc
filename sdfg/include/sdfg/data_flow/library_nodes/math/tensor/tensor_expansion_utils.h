#pragma once

#include <vector>
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::math::tensor {

// Helper structure for input access node information
struct InputContainerInfo {
    std::string name;
    bool is_const = false;
    const data_flow::Memlet* memlet;
    const data_flow::AccessNode* access_to_remove = nullptr;

    void remove_old(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block) const {
        if (memlet) {
            builder.remove_memlet(block, *memlet);
        }
        if (access_to_remove) {
            builder.remove_node(block, *access_to_remove);
        }
    }
};

// Helper structure for map dimensions
struct MapDimension {
    symbolic::Expression indvar;
    structured_control_flow::Sequence& seq;
    structured_control_flow::Map& loop;
};

// Find a usable input access node for a given connector
InputContainerInfo find_usable_input_access_node(
    data_flow::DataFlowGraph& dataflow, data_flow::LibraryNode& node, const std::string& input_conn
);

// Create a temporary variable with a given prefix and type
std::string
create_temp_var(builder::StructuredSDFGBuilder& builder, const std::string& prefix, int gen, const types::IType& type);

// Create nested maps for each dimension in the shape
std::vector<MapDimension> create_maps(
    builder::StructuredSDFGBuilder& builder,
    const std::vector<symbolic::Expression>& shape,
    structured_control_flow::Sequence& parent_seq
);

} // namespace sdfg::math::tensor
