#include "sdfg/data_flow/library_nodes/math/tensor/leaky_relu_node.h"

#include <sstream>

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace math {
namespace tensor {

LeakyReLUNode::LeakyReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_LeakyReLU, {}, {"Y", "X", "alpha"}, impl_type),
      shape_(shape) {
}

const std::vector<symbolic::Expression>& LeakyReLUNode::shape() const {
    return this->shape_;
}

void LeakyReLUNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "LeakyReLUNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }

    for (int idx : {Y_INPUT_IDX, X_INPUT_IDX}) {
        const auto& conn = this->inputs_.at(idx);
        const auto* iedge = graph.in_edge_for_connector(*this, conn);
        if (!iedge) {
            throw InvalidSDFGException("LeakyReLUNode: No memlet connected at connector: " + conn);
        }
        if (iedge->base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "LeakyReLUNode: Expected tensor type at connector '" + conn + "' but got: " + iedge->base_type().print()
            );
        }
        this->validate_shape_matches(this->shape_, static_cast<const types::Tensor&>(iedge->base_type()).layout(), conn);
    }

    const auto& alpha_conn = this->inputs_.at(ALPHA_INPUT_IDX);
    const auto* alpha_iedge = graph.in_edge_for_connector(*this, alpha_conn);
    if (!alpha_iedge) {
        throw InvalidSDFGException("LeakyReLUNode: No memlet connected at connector: " + alpha_conn);
    }
    if (alpha_iedge->base_type().type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException(
            "LeakyReLUNode: Expected scalar type at connector '" + alpha_conn +
            "' but got: " + alpha_iedge->base_type().print()
        );
    }
}

bool LeakyReLUNode::supports_integer_types() const {
    return false;
}

using Use = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome LeakyReLUNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dfg = this->get_parent();
    const auto* iedge_y = dfg.in_edge_for_connector(*this, this->inputs_.at(Y_INPUT_IDX));
    const auto* iedge_x = dfg.in_edge_for_connector(*this, this->inputs_.at(X_INPUT_IDX));
    const auto* iedge_alpha = dfg.in_edge_for_connector(*this, this->inputs_.at(ALPHA_INPUT_IDX));
    if (!iedge_y || !iedge_x || !iedge_alpha) {
        return context.unable();
    }

    auto standalone = context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead, Use::Scalar});
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& new_sequence = standalone->replace_with_sequence();
    auto [body, subset] =
        ElementWiseDataflowTensorNode::add_eltwise_scope(builder, this->debug_info_, new_sequence, this->shape_);

    types::Scalar element_type(iedge_x->base_type().primitive_type());
    auto x_container = builder.find_new_name("tmp_lrelu_x_");
    builder.add_container(x_container, element_type);
    auto pos_container = builder.find_new_name("tmp_lrelu_pos_");
    builder.add_container(pos_container, types::Scalar(types::PrimitiveType::Bool));

    // Ordered compare: NaN takes the negative branch, as in PyTorch.
    auto& load_block = builder.add_block(*body, this->debug_info_);
    {
        auto& x_in = standalone->add_indirect_read_access(load_block, X_INPUT_IDX);
        auto& x_tmp = builder.add_access(load_block, x_container, this->debug_info_);
        auto& assign =
            builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            load_block, x_in, assign, "_in", subset, iedge_x->base_type(), iedge_x->debug_info()
        );
        builder.add_computational_memlet(load_block, assign, "_out", x_tmp, {}, this->debug_info_);

        auto& zero = builder.add_constant(load_block, "0.0", element_type, this->debug_info_);
        auto& pos_tmp = builder.add_access(load_block, pos_container, this->debug_info_);
        auto& cmp =
            builder.add_tasklet(load_block, data_flow::TaskletCode::fp_ogt, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(load_block, x_tmp, cmp, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(load_block, zero, cmp, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(load_block, cmp, "_out", pos_tmp, {}, this->debug_info_);
    }

    auto& if_else = builder.add_if_else(*body, this->debug_info_);
    auto pos = symbolic::symbol(pos_container);
    auto& case_pos = builder.add_case(if_else, symbolic::Eq(pos, symbolic::__true__()), this->debug_info_);
    auto& case_neg = builder.add_case(if_else, symbolic::Eq(pos, symbolic::__false__()), this->debug_info_);

    auto& pos_block = builder.add_block(case_pos, this->debug_info_);
    {
        auto& x_tmp = builder.add_access(pos_block, x_container, this->debug_info_);
        auto& y_out = standalone->add_indirect_write_access(pos_block, Y_INPUT_IDX);
        auto& assign =
            builder.add_tasklet(pos_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(pos_block, x_tmp, assign, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(
            pos_block, assign, "_out", y_out, subset, iedge_y->base_type(), iedge_y->debug_info()
        );
    }

    auto& neg_block = builder.add_block(case_neg, this->debug_info_);
    {
        auto& x_tmp = builder.add_access(neg_block, x_container, this->debug_info_);
        auto& alpha = standalone->add_scalar_input_access(neg_block, ALPHA_INPUT_IDX);
        auto& y_out = standalone->add_indirect_write_access(neg_block, Y_INPUT_IDX);
        auto& mul =
            builder.add_tasklet(neg_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(neg_block, x_tmp, mul, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(
            neg_block, alpha, mul, "_in2", {}, iedge_alpha->base_type(), iedge_alpha->debug_info()
        );
        builder
            .add_computational_memlet(neg_block, mul, "_out", y_out, subset, iedge_y->base_type(), iedge_y->debug_info());
    }

    return standalone->successfully_expanded();
}

data_flow::PointerAccessType LeakyReLUNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    }
    return TensorNode::pointer_access_type(input_idx);
}

std::string LeakyReLUNode::toStr() const {
    std::stringstream stream;
    stream << "LeakyReLUNode(shape: ";
    TensorLayout::emit_symbolic_list(stream, this->shape_);
    stream << ")";
    return stream.str();
}

symbolic::SymbolSet LeakyReLUNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : this->shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

std::unique_ptr<data_flow::DataFlowNode> LeakyReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<
        LeakyReLUNode>(element_id, this->debug_info_, vertex, parent, this->shape_, this->implementation_type_);
}

void LeakyReLUNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void LeakyReLUNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

nlohmann::json LeakyReLUNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const LeakyReLUNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (const auto& dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& LeakyReLUNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("shape"));
    assert(j.contains("debug_info"));

    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j.at("shape")) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder.add_library_node<LeakyReLUNode>(parent, debug_info, shape);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
