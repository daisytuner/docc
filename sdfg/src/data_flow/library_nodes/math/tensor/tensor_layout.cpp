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
    ss << "TLayout(shape=[";
    for (auto& s : shape_) {
        ss << s->__str__() << ",";
    }
    ss << "], strides=[";
    for (auto& s : strides_) {
        ss << s->__str__() << ",";
    }
    ss << "])";
    return ss.str();
}

symbolic::MultiExpression TensorLayout::linear_strides(const symbolic::MultiExpression& shape) {
    symbolic::MultiExpression lin_strides;
    if (shape.empty()) {
        return lin_strides; // no shape -> no strides. Just a wrapper hiding a scalar
    }
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

symbolic::Expression TensorLayout::total_elements() const { return SymEngine::mul(shape_); }

symbolic::MultiExpression TensorLayout::linear_strides() const { return std::move(linear_strides(shape_)); }

bool TensorLayout::is_scalar() const { return shape_.empty(); }

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

bool TensorLayout::operator==(const TensorLayout& other) const {
    if (!symbolic::eq(this->offset_, other.offset_)) {
        return false;
    }

    if (this->shape_.size() != other.shape_.size()) {
        return false;
    }
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        if (!symbolic::eq(this->get_dim(i), other.get_dim(i))) {
            return false;
        }
    }

    if (this->strides_.size() != other.strides_.size()) {
        return false;
    }
    for (size_t i = 0; i < this->strides_.size(); ++i) {
        if (!symbolic::eq(this->get_stride(i), other.get_stride(i))) {
            return false;
        }
    }

    return true;
};

std::unique_ptr<TensorLayout> TensorLayout::newaxis(size_t axis) const {
    if (axis > this->shape_.size()) {
        throw std::out_of_range("axis out of range for newaxis");
    }

    symbolic::MultiExpression new_shape = this->shape_;
    symbolic::MultiExpression new_strides = this->strides_;

    new_shape.insert(new_shape.begin() + axis, SymEngine::integer(1));
    new_strides.insert(new_strides.begin() + axis, SymEngine::integer(0));

    return std::make_unique<TensorLayout>(new_shape, new_strides, this->offset_);
}

std::unique_ptr<TensorLayout> TensorLayout::flip(size_t axis) const {
    if (axis >= shape_.size()) {
        throw std::out_of_range("axis out of range for flip");
    }

    symbolic::MultiExpression new_strides = this->strides_;

    // Negate the stride for the specified axis
    new_strides[axis] = SymEngine::neg(this->strides_[axis]);

    // Compute new offset: offset += stride * (shape - 1)
    auto shape_minus_one = SymEngine::sub(this->shape_[axis], SymEngine::integer(1));
    auto offset_adjustment = SymEngine::mul(this->strides_[axis], shape_minus_one);

    symbolic::Expression new_offset = SymEngine::add(this->offset_, offset_adjustment);

    return std::make_unique<TensorLayout>(this->shape_, new_strides, new_offset);
}

std::unique_ptr<TensorLayout> TensorLayout::unsqueeze(size_t axis) const { return this->newaxis(axis); }

std::unique_ptr<TensorLayout> TensorLayout::squeeze(size_t axis) const {
    if (axis >= this->shape_.size()) {
        throw std::out_of_range("axis out of range for squeeze");
    }

    if (!SymEngine::is_a<SymEngine::Integer>(*this->shape_.at(axis))) {
        throw std::invalid_argument("cannot squeeze axis with symbolic size");
    }
    auto dim_size = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(this->shape_.at(axis))->as_int();
    if (dim_size != 1) {
        throw std::invalid_argument("cannot squeeze axis with size != 1");
    }

    symbolic::MultiExpression new_shape = this->shape_;
    symbolic::MultiExpression new_strides = this->strides_;

    new_shape.erase(new_shape.begin() + axis);
    new_strides.erase(new_strides.begin() + axis);

    return std::make_unique<TensorLayout>(new_shape, new_strides, this->offset_);
}

std::unique_ptr<TensorLayout> TensorLayout::squeeze() const {
    symbolic::MultiExpression new_shape;
    symbolic::MultiExpression new_strides;

    for (size_t i = 0; i < this->shape_.size(); ++i) {
        bool is_size_one = false;
        if (SymEngine::is_a<SymEngine::Integer>(*this->shape_.at(i))) {
            auto dim_size = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(this->shape_.at(i))->as_int();
            is_size_one = (dim_size == 1);
        }

        if (!is_size_one) {
            new_shape.push_back(this->shape_.at(i));
            new_strides.push_back(this->strides_.at(i));
        }
    }

    return std::make_unique<TensorLayout>(new_shape, new_strides, this->offset_);
}

std::unique_ptr<TensorLayout> TensorLayout::reshape(const symbolic::MultiExpression& new_shape) const {
    // Compute the total number of elements in the current shape
    symbolic::Expression total_elements = this->total_elements();

    // Compute the total number of elements in the new shape
    symbolic::Expression new_total_elements = symbolic::one();
    for (const auto& dim : new_shape) {
        new_total_elements = symbolic::mul(new_total_elements, dim);
    }

    // Check if the total number of elements matches
    if (!symbolic::eq(total_elements, new_total_elements)) {
        throw std::invalid_argument("total number of elements must match for reshape");
    }

    // Compute new strides based on the new shape
    symbolic::MultiExpression new_strides = linear_strides(new_shape);

    return std::make_unique<TensorLayout>(new_shape, new_strides, offset_);
}

} // namespace sdfg::math::tensor
