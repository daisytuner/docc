#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"

#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

// -----------------------------------------------------------------------------
// FFTNodeBase
// -----------------------------------------------------------------------------

FFTNodeBase::FFTNodeBase(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const data_flow::ImplementationType& implementation_type,
    const std::vector<symbolic::Expression>& shape,
    symbolic::Expression batch,
    types::PrimitiveType precision
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          code,
          {}, // outputs (the result buffer `__Y` is a pointer input)
          {"__Y", "__X"},
          implementation_type,
          true // side effects: writes to the output buffer
      ),
      shape_(shape), batch_(std::move(batch)), precision_(precision) {}

const std::vector<symbolic::Expression>& FFTNodeBase::shape() const { return this->shape_; }

symbolic::Expression FFTNodeBase::batch() const { return this->batch_; }

size_t FFTNodeBase::rank() const { return this->shape_.size(); }

types::PrimitiveType FFTNodeBase::real_primitive() const { return this->precision_; }

types::PrimitiveType FFTNodeBase::complex_primitive() const {
    switch (this->precision_) {
        case types::PrimitiveType::Float:
            return types::PrimitiveType::CFloat;
        case types::PrimitiveType::Double:
            return types::PrimitiveType::CDouble;
        default:
            return types::PrimitiveType::Void;
    }
}

symbolic::Expression FFTNodeBase::complex_last_dim() const {
    // Hermitian layout: last transformed dimension is reduced to n/2 + 1.
    const auto& last = this->shape_.back();
    return symbolic::add(symbolic::div(last, symbolic::integer(2)), symbolic::one());
}

symbolic::Expression FFTNodeBase::real_extent() const {
    symbolic::Expression extent = this->batch_;
    for (const auto& dim : this->shape_) {
        extent = symbolic::mul(extent, dim);
    }
    return extent;
}

symbolic::Expression FFTNodeBase::complex_extent() const {
    symbolic::Expression extent = this->batch_;
    for (size_t i = 0; i + 1 < this->shape_.size(); ++i) {
        extent = symbolic::mul(extent, this->shape_[i]);
    }
    return symbolic::mul(extent, this->complex_last_dim());
}

void FFTNodeBase::validate(const Function& function) const {
    data_flow::CodeNode::validate(function);

    if (this->shape_.empty()) {
        throw InvalidSDFGException("FFTNode: shape must have at least one dimension");
    }
    if (this->precision_ != types::PrimitiveType::Float && this->precision_ != types::PrimitiveType::Double) {
        throw InvalidSDFGException("FFTNode: precision must be Float or Double");
    }
}

symbolic::SymbolSet FFTNodeBase::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : this->shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (auto& atom : symbolic::atoms(this->batch_)) {
        syms.insert(atom);
    }
    return syms;
}

void FFTNodeBase::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    this->batch_ = symbolic::subs(this->batch_, old_expression, new_expression);
}

void FFTNodeBase::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    this->batch_ = symbolic::subs(this->batch_, replacements);
}

data_flow::PointerAccessType FFTNodeBase::pointer_access_type(int input_idx) const {
    // Forward: __X real (real_extent), __Y complex (complex_extent).
    // Inverse: __X complex (complex_extent), __Y real (real_extent).
    const bool forward = this->direction() == FFTDirection::Forward;
    if (input_idx == Y_INPUT_IDX) {
        auto range = forward ? this->complex_extent() : this->real_extent();
        return data_flow::PointerAccessMeta::create_full_write_only(range, true);
    } else if (input_idx == X_INPUT_IDX) {
        auto range = forward ? this->real_extent() : this->complex_extent();
        return data_flow::PointerAccessMeta::create_read_only(range, true);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

// -----------------------------------------------------------------------------
// FFTNode (forward, R2C)
// -----------------------------------------------------------------------------

FFTNode::FFTNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const std::vector<symbolic::Expression>& shape,
    symbolic::Expression batch,
    types::PrimitiveType precision
)
    : FFTNodeBase(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_FFT,
          implementation_type,
          shape,
          std::move(batch),
          precision
      ) {}

passes::LibNodeExpander::ExpandOutcome FFTNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    // TODO(fft): naive DFT reference expansion (CPU-correct) for the
    // ImplementationType_NONE path. Until then the node is only realizable via
    // the CUDA (cuFFT) / ROCm (hipFFT) dispatchers selected by the offloading
    // rewriter passes.
    return context.unable();
}

std::unique_ptr<data_flow::DataFlowNode> FFTNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<FFTNode>(new FFTNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->shape_,
        this->batch_,
        this->precision_
    ));
}

std::string FFTNode::toStr() const {
    std::string s = LibraryNode::toStr() + "(R2C, batch=" + this->batch_->__str__() + ", shape=[";
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        s += this->shape_[i]->__str__();
        if (i + 1 < this->shape_.size()) {
            s += ", ";
        }
    }
    s += "])";
    return s;
}

// -----------------------------------------------------------------------------
// IFFTNode (inverse, C2R)
// -----------------------------------------------------------------------------

IFFTNode::IFFTNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const std::vector<symbolic::Expression>& shape,
    symbolic::Expression batch,
    types::PrimitiveType precision
)
    : FFTNodeBase(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_IFFT,
          implementation_type,
          shape,
          std::move(batch),
          precision
      ) {}

passes::LibNodeExpander::ExpandOutcome IFFTNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    // TODO(fft): naive inverse DFT reference expansion (CPU-correct). See FFTNode::expand.
    return context.unable();
}

std::unique_ptr<data_flow::DataFlowNode> IFFTNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<IFFTNode>(new IFFTNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->shape_,
        this->batch_,
        this->precision_
    ));
}

std::string IFFTNode::toStr() const {
    std::string s = LibraryNode::toStr() + "(C2R, batch=" + this->batch_->__str__() + ", shape=[";
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        s += this->shape_[i]->__str__();
        if (i + 1 < this->shape_.size()) {
            s += ", ";
        }
    }
    s += "])";
    return s;
}

// -----------------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------------

namespace {

nlohmann::json serialize_fft(const FFTNodeBase& node) {
    nlohmann::json j;
    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();

    nlohmann::json shape = nlohmann::json::array();
    for (const auto& dim : node.shape()) {
        shape.push_back(serializer.expression(dim));
    }
    j["shape"] = shape;
    j["batch"] = serializer.expression(node.batch());
    j["precision"] = types::primitive_type_to_string(node.real_primitive());
    return j;
}

std::vector<symbolic::Expression> parse_shape(const nlohmann::json& j) {
    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j.at("shape")) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    return shape;
}

} // namespace

nlohmann::json FFTNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    return serialize_fft(static_cast<const FFTNode&>(library_node));
}

data_flow::LibraryNode& FFTNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    if (j["code"].get<std::string>() != LibraryNodeType_FFT.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto shape = parse_shape(j);
    auto batch = symbolic::parse(j.at("batch").get<std::string>());
    auto precision = types::primitive_type_from_string(j.at("precision").get<std::string>());
    data_flow::ImplementationType implementation_type(j.at("implementation_type").get<std::string>());

    return builder.add_library_node<FFTNode>(parent, debug_info, implementation_type, shape, batch, precision);
}

nlohmann::json IFFTNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    return serialize_fft(static_cast<const IFFTNode&>(library_node));
}

data_flow::LibraryNode& IFFTNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    if (j["code"].get<std::string>() != LibraryNodeType_IFFT.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto shape = parse_shape(j);
    auto batch = symbolic::parse(j.at("batch").get<std::string>());
    auto precision = types::primitive_type_from_string(j.at("precision").get<std::string>());
    data_flow::ImplementationType implementation_type(j.at("implementation_type").get<std::string>());

    return builder.add_library_node<IFFTNode>(parent, debug_info, implementation_type, shape, batch, precision);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
