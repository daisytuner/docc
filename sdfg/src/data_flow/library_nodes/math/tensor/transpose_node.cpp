#include "sdfg/data_flow/library_nodes/math/tensor/transpose_node.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace tensor {

TransposeNode::TransposeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& perm
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Transpose,
          {"Y"},
          {"X"},
          data_flow::ImplementationType_NONE
      ),
      shape_(shape), perm_(perm) {
    if (perm_.empty()) {
        // Default permutation: reverse
        for (size_t i = 0; i < shape.size(); ++i) {
            perm_.push_back(shape.size() - 1 - i);
        }
    } else {
        if (perm_.size() != shape_.size()) {
            throw std::invalid_argument("Permutation rank must match shape rank");
        }
    }
}

void TransposeNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    auto& iedge = *graph.in_edges(*this).begin();
    auto& shape = static_cast<const types::Tensor&>(iedge.base_type());
    if (shape.shape().size() != this->shape_.size()) {
        throw InvalidSDFGException(
            "Library Node: Tensor shape must match node shape. Tensor shape: " + std::to_string(shape.shape().size()) +
            " Node shape: " + std::to_string(this->shape_.size())
        );
    }
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        if (!symbolic::eq(shape.shape().at(i), this->shape_.at(i))) {
            throw InvalidSDFGException(
                "Library Node: Tensor shape does not match expected shape. Tensor shape: " +
                shape.shape().at(i)->__str__() + " Expected shape: " + this->shape_.at(i)->__str__()
            );
        }
    }

    auto& oedge = *graph.out_edges(*this).begin();
    auto& output_shape = static_cast<const types::Tensor&>(oedge.base_type());
    if (output_shape.shape().size() != this->shape_.size()) {
        throw InvalidSDFGException(
            "Library Node: Output tensor shape must match node shape. Output tensor shape: " +
            std::to_string(output_shape.shape().size()) + " Node shape: " + std::to_string(this->shape_.size())
        );
    }

    for (size_t i = 0; i < this->shape_.size(); ++i) {
        if (!symbolic::eq(output_shape.shape().at(i), this->shape_.at(perm_.at(i)))) {
            throw InvalidSDFGException(
                "Library Node: Output tensor shape does not match expected shape. Output tensor shape: " +
                output_shape.shape().at(i)->__str__() + " Expected shape: " + this->shape_.at(perm_[i])->__str__()
            );
        }
    }
}

symbolic::SymbolSet TransposeNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void TransposeNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

bool TransposeNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    for (size_t i = 0; i < this->shape_.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, this->shape_[i]);
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        loop_vars.push_back(indvar);
    }

    auto& body = builder.add_block(*last_scope, {}, block.debug_info());

    // Determine output shape
    std::vector<symbolic::Expression> output_shape(shape_.size());
    std::vector<symbolic::Expression> output_indices(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
        output_shape[i] = shape_[perm_[i]];
        output_indices[i] = loop_vars[perm_[i]];
    }

    // Read Input
    auto& x_access = builder.add_access(body, input_node.data(), debug_info());
    auto& y_access = builder.add_access(body, output_node.data(), debug_info());

    auto& tasklet = builder.add_tasklet(body, data_flow::assign, "_out", {"_in"}, debug_info());

    // Access memlets
    builder.add_computational_memlet(body, x_access, tasklet, "_in", loop_vars, iedge.base_type(), debug_info());

    builder.add_computational_memlet(body, tasklet, "_out", y_access, output_indices, oedge.base_type(), debug_info());

    // Remove the original node
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

nlohmann::json TransposeNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const TransposeNode& transpose_node = static_cast<const TransposeNode&>(library_node);
    nlohmann::json j;

    j["code"] = transpose_node.code().value();

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : transpose_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["perm"] = nlohmann::json::array();
    for (auto& dim : transpose_node.perm()) {
        j["perm"].push_back(dim);
    }

    return j;
}

data_flow::LibraryNode& TransposeNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("shape"));
    assert(j.contains("perm"));

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    std::vector<int64_t> perm;
    for (const auto& dim : j["perm"]) {
        perm.push_back(dim.get<int64_t>());
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<TransposeNode>(parent, debug_info, shape, perm);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
