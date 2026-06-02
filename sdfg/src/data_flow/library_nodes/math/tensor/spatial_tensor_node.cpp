#include "sdfg/data_flow/library_nodes/math/tensor/spatial_tensor_node.h"

namespace sdfg::math::tensor {


SpatialTensorNode::SpatialTensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const data_flow::ImplementationType& impl_type,
    QuantizationType quantization,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& strides,
    const std::vector<symbolic::Expression>& pads,
    const std::vector<symbolic::Expression>& dilations
)
    : TensorNode(element_id, debug_info, vertex, parent, code, outputs, inputs, impl_type), shape_(shape),
      kernel_shape_(kernel_shape), strides_(strides), pads_(pads), dilations_(dilations),
      fixed_quantization_(quantization) {}

QuantizationType SpatialTensorNode::fixed_quantization() const { return fixed_quantization_; }

QuantizationType SpatialTensorNode::quantization(const data_flow::DataFlowGraph& data_flow_graph) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        return fixed_quantization_;
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

std::optional<QuantizationType> SpatialTensorNode::uniform_quantization(const data_flow::DataFlowGraph& data_flow_graph
) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        auto inferred = this->primitive_type(data_flow_graph);
        if (inferred == fixed_quantization_) {
            return fixed_quantization_;
        } else {
            return std::nullopt;
        }
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

void SpatialTensorNode::set_fixed_quantization(const QuantizationType quant) { fixed_quantization_ = quant; }

symbolic::SymbolSet SpatialTensorNode::symbols() const {
    symbolic::SymbolSet syms;
    for (auto& expr : shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : kernel_shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : strides_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : pads_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : dilations_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void SpatialTensorNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& expr : shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : kernel_shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : strides_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : pads_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : dilations_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
}

size_t SpatialTensorNode::num_spatial_dims() const {
    auto& s = shape_;
    assert(s.size() >= 2);
    return s.size() - 2;
}

symbolic::Expression SpatialTensorNode::output_spatial_dim(size_t i) const {
    size_t n_spatial = num_spatial_dims();
    assert(i < n_spatial);

    auto& s = shape_;
    auto& ks = kernel_shape_;
    auto& st = strides_;
    auto& pa = pads_;
    auto& di = dilations_;

    auto d_in = s[2 + i];
    auto k = ks[i];

    symbolic::Expression stride = st.empty() ? symbolic::Expression(symbolic::one()) : st[i];
    symbolic::Expression dilation = di.empty() ? symbolic::Expression(symbolic::one()) : di[i];

    // pads layout: [begin_d0, begin_d1, …, end_d0, end_d1, …]
    symbolic::Expression pad_begin = pa.empty() ? symbolic::Expression(symbolic::zero()) : pa[i];
    symbolic::Expression pad_end = pa.empty() ? symbolic::Expression(symbolic::zero()) : pa[n_spatial + i];

    // numerator = D_i + pad_begin + pad_end - dilation * (k - 1) - 1
    auto numerator = symbolic::
        sub(symbolic::add(symbolic::add(d_in, pad_begin), pad_end),
            symbolic::add(symbolic::mul(dilation, symbolic::sub(k, symbolic::one())), symbolic::one()));

    return symbolic::add(symbolic::div(numerator, stride), symbolic::one());
}

symbolic::Expression SpatialTensorNode::output_spatial_volume() const {
    size_t n_spatial = num_spatial_dims();
    symbolic::Expression result = symbolic::Expression(symbolic::one());
    for (size_t i = 0; i < n_spatial; ++i) {
        result = symbolic::mul(result, output_spatial_dim(i));
    }
    return result;
}

symbolic::Expression SpatialTensorNode::kernel_volume() const {
    auto& ks = kernel_shape_;
    return SymEngine::mul(ks);
}

std::basic_ostream<char>& SpatialTensorNode::operator<<(std::basic_ostream<char>& os) const {
    os << "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) os << ", ";
        os << shape_[i]->__str__();
    }
    os << "], kernel_shape=[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
        if (i > 0) os << ", ";
        os << kernel_shape_[i]->__str__();
    }
    os << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) os << ", ";
        os << strides_[i]->__str__();
    }
    os << "], pads=[";
    for (size_t i = 0; i < pads_.size(); ++i) {
        if (i > 0) os << ", ";
        os << pads_[i]->__str__();
    }
    os << "], dilations=[";
    for (size_t i = 0; i < dilations_.size(); ++i) {
        if (i > 0) os << ", ";
        os << dilations_[i]->__str__();
    }
    os << "], ";
    os << "quant=" << types::primitive_type_to_string(fixed_quantization_);
    return os;
}

void SpatialTensorNodeBaseSerializer::fill_base_values(const SpatialTensorNode& node, nlohmann::json& j) {
    j["code"] = node.code().value();

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    j["result_quant"] = node.quantization();
}

SpatialTensorNodeBaseSerializer::BaseDeser SpatialTensorNodeBaseSerializer::deserialize_base_values(const nlohmann::json&
                                                                                                        j) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("kernel_shape"));

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> kernel_shape;
    for (const auto& dim : j["kernel_shape"]) {
        kernel_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    std::vector<symbolic::Expression> strides;
    if (j.contains("strides")) {
        for (const auto& stride : j["strides"]) {
            strides.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> pads;
    if (j.contains("pads")) {
        for (const auto& pad : j["pads"]) {
            pads.push_back(symbolic::parse(pad.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> dilations;
    if (j.contains("dilations")) {
        for (const auto& dilation : j["dilations"]) {
            dilations.push_back(symbolic::parse(dilation.get<std::string>()));
        }
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return {
        .shape = shape,
        .kernel_shape = kernel_shape,
        .strides = strides,
        .pads = pads,
        .dilations = dilations,
        .quantization = deserialize_quantization(j, "result_quant", QUANTIZATION_MATCH_INPUTS),
        .debug_info = debug_info
    };
}

} // namespace sdfg::math::tensor
