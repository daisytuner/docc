#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg::math::tensor {

TensorLayout::TensorLayout(
    const symbolic::MultiExpression& shape, const symbolic::MultiExpression& strides, const symbolic::Expression offset
)
    : shape_(shape), strides_(strides), offset_(offset) {
    if (strides.empty()) {
        strides_ = linear_strides();
    }
}

void TensorLayout::serialize_to_json(nlohmann::json& j) const {
    nlohmann::json shape_arr = nlohmann::json::array();
    sdfg::serializer::JSONSerializer serializer;

    for (auto& dim : shape_) {
        shape_arr.push_back(serializer.expression(dim));
    }
    j["shape"] = shape_arr;

    nlohmann::json stride_arr = nlohmann::json::array();
    for (auto& stride : strides_) {
        stride_arr.push_back(serializer.expression(stride));
    }
    j["strides"] = stride_arr;

    j["offset"] = serializer.expression(offset_);
}

std::string TensorLayout::toStr() const {
    std::stringstream ss;
    ss << "TLayout(" << this << ")";
    return ss.str();
}

symbolic::MultiExpression TensorLayout::linear_strides(const symbolic::MultiExpression& shape) {
    symbolic::MultiExpression lin_strides;
    std::size_t dims = shape.size();
    lin_strides.resize(dims);
    lin_strides[dims - 1] = symbolic::integer(1);
    for (int i = static_cast<int>(dims) - 2; i >= 0; --i) {
        lin_strides[i] = symbolic::mul(lin_strides.at(i + 1), shape.at(i + 1));
    }

    return std::move(lin_strides);
}
void TensorLayout::collect_symbols(symbolic::SymbolSet& syms) const {
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : strides_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (auto& atom : symbolic::atoms(offset_)) {
        syms.insert(atom);
    }
}

void TensorLayout::replace_symbols(const symbolic::Expression& old, const symbolic::Expression& new_expr) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old, new_expr);
    }
    for (auto& stride : strides_) {
        stride = symbolic::subs(stride, old, new_expr);
    }
    offset_ = symbolic::subs(offset_, old, new_expr);
}

symbolic::MultiExpression TensorLayout::linear_strides() const { return std::move(linear_strides(shape_)); }

TensorLayout TensorLayout::deserialize_from_json(const nlohmann::json& j) {
    symbolic::MultiExpression shape;
    for (const auto& dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    symbolic::MultiExpression strides;
    for (const auto& stride : j["strides"]) {
        strides.push_back(symbolic::parse(stride.get<std::string>()));
    }

    symbolic::Expression offset = symbolic::parse(j["offset"].get<std::string>());

    return std::move(TensorLayout(shape, strides, offset));
}

std::ostream& operator<<(std::ostream& stream, const TensorLayout& layout) {
    stream << "{shape[";
    for (size_t i = 0; i < layout.shape().size(); ++i) {
        if (i > 0) stream << ", ";
        stream << layout.shape().at(i)->__str__();
    }
    stream << "], strides=[";
    for (size_t i = 0; i < layout.strides().size(); ++i) {
        if (i > 0) stream << ", ";
        stream << layout.strides().at(i)->__str__();
    }
    stream << "]";
    if (SymEngine::neq(*layout.offset(), *symbolic::integer(0))) {
        stream << ", off=" << layout.offset()->__str__();
    }
    stream << "}";

    return stream;
}

bool TensorLayout::has_linear_accesses_no_padding(symbolic::MultiExpression shape, symbolic::MultiExpression strides) {
    auto basic_strides = types::Tensor::strides_from_shape(shape);
    if (basic_strides.size() != strides.size()) {
        return false;
    }
    for (size_t i = 0; i < strides.size(); i++) {
        if (!symbolic::eq(basic_strides.at(i), strides.at(i))) {
            return false;
        }
    }
    return true;
}

bool TensorLayout::has_linear_accesses_no_padding() const { return has_linear_accesses_no_padding(shape_, strides_); }

bool TensorLayout::has_transposed_strides_no_padding() const {
    if (shape_.size() < 2) {
        return false;
    }
    symbolic::MultiExpression new_shape;
    new_shape.reserve(shape_.size());
    for (size_t i = 0; i < shape_.size() - 2; i++) {
        new_shape.push_back(shape_.at(i));
    }
    new_shape.push_back(shape_.at(shape_.size() - 1));
    new_shape.push_back(shape_.at(shape_.size() - 2));
    symbolic::MultiExpression transposed_strides(strides_);
    transposed_strides[strides_.size() - 2] = strides_.at(strides_.size() - 1);
    transposed_strides[strides_.size() - 1] = strides_.at(strides_.size() - 2);
    return TensorLayout::has_linear_accesses_no_padding(new_shape, transposed_strides);
}

} // namespace sdfg::math::tensor
