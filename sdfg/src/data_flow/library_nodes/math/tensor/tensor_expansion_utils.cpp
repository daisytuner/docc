#include "sdfg/data_flow/library_nodes/math/tensor/tensor_expansion_utils.h"

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/types/scalar.h"

namespace sdfg::math::tensor {

InputContainerInfo find_usable_input_access_node(
    data_flow::DataFlowGraph& dataflow, data_flow::LibraryNode& node, const std::string& input_conn
) {
    auto* edge = dataflow.in_edge_for_connector(node, input_conn);
    if (!edge) {
        throw InvalidSDFGException(node.toStr() + " requires input on " + input_conn);
    }
    auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&edge->src());
    if (!access_node) {
        throw InvalidSDFGException(node.toStr() + " requires input on " + input_conn + " to be an access node");
    }

    return {
        .name = access_node->data(),
        .is_const = !!dynamic_cast<const data_flow::ConstantNode*>(&edge->src()),
        .memlet = edge,
        .access_to_remove = access_node
    };
}

std::string
create_temp_var(builder::StructuredSDFGBuilder& builder, const std::string& prefix, int gen, const types::IType& type) {
    std::string n = prefix + "_" + std::to_string(gen);
    auto name = builder.find_new_name(n);
    builder.add_container(name, type);
    return name;
}

std::vector<MapDimension> create_maps(
    builder::StructuredSDFGBuilder& builder,
    const std::vector<symbolic::Expression>& shape,
    structured_control_flow::Sequence& parent_seq
) {
    std::vector<MapDimension> result;
    result.reserve(shape.size());

    types::Scalar indvar_type(types::PrimitiveType::Int64);
    structured_control_flow::Sequence* current_seq = &parent_seq;

    for (size_t i = 0; i < shape.size(); ++i) {
        // Create induction variable for this dimension
        std::string loop_var_name = builder.find_new_name("i" + std::to_string(i));
        builder.add_container(loop_var_name, indvar_type);
        auto loop_var = symbolic::symbol(loop_var_name);

        // Create the map: for (i = 0; i < dim_size; i++)
        auto& map = builder.add_map(
            *current_seq,
            loop_var,
            symbolic::Lt(loop_var, shape[i]),
            symbolic::integer(0),
            symbolic::add(loop_var, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        // Store the map dimension info
        result.push_back(MapDimension{.indvar = loop_var, .seq = map.root(), .loop = map});

        // Next iteration will add to the sequence inside this map
        current_seq = &map.root();
    }

    return result;
}

} // namespace sdfg::math::tensor
