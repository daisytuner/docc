#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"

#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

namespace {

std::vector<std::string> make_inputs(bool with_bias) {
    std::vector<std::string> inputs{"Y", "X", "W"};
    if (with_bias) {
        inputs.push_back("B");
    }
    return inputs;
}

} // namespace

FFTConvNode::FFTConvNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& pads,
    types::PrimitiveType precision,
    bool with_bias
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_FFTConv,
          {}, // outputs (Y is a pointer input)
          make_inputs(with_bias),
          implementation_type,
          true // side effects: writes to the output buffer
      ),
      shape_(shape), kernel_shape_(kernel_shape), pads_(pads), precision_(precision), with_bias_(with_bias) {}

const std::vector<symbolic::Expression>& FFTConvNode::shape() const { return this->shape_; }

const std::vector<symbolic::Expression>& FFTConvNode::kernel_shape() const { return this->kernel_shape_; }

const std::vector<symbolic::Expression>& FFTConvNode::pads() const { return this->pads_; }

types::PrimitiveType FFTConvNode::real_primitive() const { return this->precision_; }

types::PrimitiveType FFTConvNode::complex_primitive() const {
    switch (this->precision_) {
        case types::PrimitiveType::Float:
            return types::PrimitiveType::CFloat;
        case types::PrimitiveType::Double:
            return types::PrimitiveType::CDouble;
        default:
            return types::PrimitiveType::Void;
    }
}

bool FFTConvNode::with_bias() const { return this->with_bias_; }

void FFTConvNode::validate(const Function& function) const {
    data_flow::CodeNode::validate(function);

    if (this->shape_.size() != 4) {
        throw InvalidSDFGException("FFTConvNode: shape must be [N, C, H, W]");
    }
    if (this->kernel_shape_.size() != 2) {
        throw InvalidSDFGException("FFTConvNode: kernel_shape must be [Kh, Kw]");
    }
    if (this->precision_ != types::PrimitiveType::Float && this->precision_ != types::PrimitiveType::Double) {
        throw InvalidSDFGException("FFTConvNode: precision must be Float or Double");
    }
}

symbolic::SymbolSet FFTConvNode::symbols() const {
    symbolic::SymbolSet syms;
    auto collect = [&syms](const std::vector<symbolic::Expression>& v) {
        for (const auto& e : v) {
            for (auto& atom : symbolic::atoms(e)) {
                syms.insert(atom);
            }
        }
    };
    collect(this->shape_);
    collect(this->kernel_shape_);
    collect(this->pads_);
    return syms;
}

void FFTConvNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    auto subs_all = [&](std::vector<symbolic::Expression>& v) {
        for (auto& e : v) {
            e = symbolic::subs(e, old_expression, new_expression);
        }
    };
    subs_all(this->shape_);
    subs_all(this->kernel_shape_);
    subs_all(this->pads_);
}

void FFTConvNode::replace(const symbolic::ExpressionMapping& replacements) {
    auto subs_all = [&](std::vector<symbolic::Expression>& v) {
        for (auto& e : v) {
            e = symbolic::subs(e, replacements);
        }
    };
    subs_all(this->shape_);
    subs_all(this->kernel_shape_);
    subs_all(this->pads_);
}

passes::LibNodeExpander::ExpandOutcome FFTConvNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    // Fused GPU-only node: realized by the CUDA/ROCm hand-tuned dispatchers, never expanded.
    return context.unable();
}

std::unique_ptr<data_flow::DataFlowNode> FFTConvNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<FFTConvNode>(new FFTConvNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->shape_,
        this->kernel_shape_,
        this->pads_,
        this->precision_,
        this->with_bias_
    ));
}

std::string FFTConvNode::toStr() const {
    return LibraryNode::toStr() + "(fft_conv, " + (this->precision_ == types::PrimitiveType::Double ? "d" : "s") +
           (this->with_bias_ ? ", bias" : "") + ")";
}

data_flow::PointerAccessType FFTConvNode::pointer_access_type(int input_idx) const {
    // The hand-tuned dispatcher manages its own device buffers, but declare intent for analyses.
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(SymEngine::null, true);
    } else if (input_idx == X_INPUT_IDX || input_idx == W_INPUT_IDX || input_idx == B_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(SymEngine::null, true);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

// -----------------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------------

namespace {

nlohmann::json serialize_vec(serializer::JSONSerializer& s, const std::vector<symbolic::Expression>& v) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& e : v) {
        arr.push_back(s.expression(e));
    }
    return arr;
}

std::vector<symbolic::Expression> parse_vec(const nlohmann::json& arr) {
    std::vector<symbolic::Expression> v;
    for (const auto& e : arr) {
        v.push_back(symbolic::parse(e.get<std::string>()));
    }
    return v;
}

} // namespace

nlohmann::json FFTConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const FFTConvNode&>(library_node);
    nlohmann::json j;
    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    j["shape"] = serialize_vec(serializer, node.shape());
    j["kernel_shape"] = serialize_vec(serializer, node.kernel_shape());
    j["pads"] = serialize_vec(serializer, node.pads());
    j["precision"] = types::primitive_type_to_string(node.real_primitive());
    j["with_bias"] = node.with_bias();
    return j;
}

data_flow::LibraryNode& FFTConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    if (j["code"].get<std::string>() != LibraryNodeType_FFTConv.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto shape = parse_vec(j.at("shape"));
    auto kernel_shape = parse_vec(j.at("kernel_shape"));
    auto pads = parse_vec(j.at("pads"));
    auto precision = types::primitive_type_from_string(j.at("precision").get<std::string>());
    bool with_bias = j.at("with_bias").get<bool>();
    data_flow::ImplementationType implementation_type(j.at("implementation_type").get<std::string>());

    return builder.add_library_node<
        FFTConvNode>(parent, debug_info, implementation_type, shape, kernel_shape, pads, precision, with_bias);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
