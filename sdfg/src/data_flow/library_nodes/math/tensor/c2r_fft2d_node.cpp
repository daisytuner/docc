#include "sdfg/data_flow/library_nodes/math/tensor/c2r_fft2d_node.h"

#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

C2RFFT2DNode::C2RFFT2DNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const std::vector<symbolic::Expression>& shape,
    types::PrimitiveType precision
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_C2RFFT2D,
          {}, // outputs (Y is a pointer input)
          {"Y", "X"},
          implementation_type,
          true // side effects: writes to the output buffer
      ),
      shape_(shape), precision_(precision) {}

const std::vector<symbolic::Expression>& C2RFFT2DNode::shape() const { return this->shape_; }

types::PrimitiveType C2RFFT2DNode::real_primitive() const { return this->precision_; }

types::PrimitiveType C2RFFT2DNode::complex_primitive() const {
    switch (this->precision_) {
        case types::PrimitiveType::Float:
            return types::PrimitiveType::CFloat;
        case types::PrimitiveType::Double:
            return types::PrimitiveType::CDouble;
        default:
            return types::PrimitiveType::Void;
    }
}

void C2RFFT2DNode::validate(const Function& function) const {
    data_flow::CodeNode::validate(function);

    if (this->shape_.size() != 3) {
        throw InvalidSDFGException("C2RFFT2DNode: shape must be [matrices, fftH, fftW]");
    }
    if (this->precision_ != types::PrimitiveType::Float && this->precision_ != types::PrimitiveType::Double) {
        throw InvalidSDFGException("C2RFFT2DNode: precision must be Float or Double");
    }
}

symbolic::SymbolSet C2RFFT2DNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& e : this->shape_) {
        for (auto& atom : symbolic::atoms(e)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void C2RFFT2DNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& e : this->shape_) {
        e = symbolic::subs(e, old_expression, new_expression);
    }
}

void C2RFFT2DNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& e : this->shape_) {
        e = symbolic::subs(e, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome C2RFFT2DNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    // GPU-only node: realized by the CUDA/ROCm hand-tuned dispatchers, never expanded.
    return context.unable();
}

std::unique_ptr<data_flow::DataFlowNode> C2RFFT2DNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<C2RFFT2DNode>(new C2RFFT2DNode(
        element_id, this->debug_info(), vertex, parent, this->implementation_type_, this->shape_, this->precision_
    ));
}

std::string C2RFFT2DNode::toStr() const {
    return LibraryNode::toStr() + "(c2r_fft2d, " + (this->precision_ == types::PrimitiveType::Double ? "d" : "s") + ")";
}

data_flow::PointerAccessType C2RFFT2DNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(SymEngine::null, true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(SymEngine::null, true);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

// -----------------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------------

nlohmann::json C2RFFT2DNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const C2RFFT2DNode&>(library_node);
    nlohmann::json j;
    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    nlohmann::json shape = nlohmann::json::array();
    for (const auto& e : node.shape()) {
        shape.push_back(serializer.expression(e));
    }
    j["shape"] = shape;
    j["precision"] = types::primitive_type_to_string(node.real_primitive());
    return j;
}

data_flow::LibraryNode& C2RFFT2DNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    if (j["code"].get<std::string>() != LibraryNodeType_C2RFFT2D.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::vector<symbolic::Expression> shape;
    for (const auto& e : j.at("shape")) {
        shape.push_back(symbolic::parse(e.get<std::string>()));
    }
    auto precision = types::primitive_type_from_string(j.at("precision").get<std::string>());
    data_flow::ImplementationType implementation_type(j.at("implementation_type").get<std::string>());

    return builder.add_library_node<C2RFFT2DNode>(parent, debug_info, implementation_type, shape, precision);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
