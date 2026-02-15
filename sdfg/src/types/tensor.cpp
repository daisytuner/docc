#include "sdfg/types/tensor.h"

#include "sdfg/types/scalar.h"

namespace sdfg {
namespace types {

symbolic::MultiExpression Tensor::strides_from_shape(const symbolic::MultiExpression& shape) {
    if (shape.empty()) {
        return {};
    }
    symbolic::MultiExpression strides(shape.size());
    strides.back() = SymEngine::integer(1);
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = SymEngine::mul(strides[i], shape[i]);
    }
    return strides;
}

Tensor::Tensor(const Scalar& element_type, const symbolic::MultiExpression& shape)
    : element_type_(std::unique_ptr<Scalar>(static_cast<Scalar*>(element_type.clone().release()))), shape_(shape),
      strides_(strides_from_shape(shape)), offset_(symbolic::zero()) {};

Tensor::Tensor(
    const Scalar& element_type,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& strides,
    const symbolic::Expression& offset
)
    : element_type_(std::unique_ptr<Scalar>(static_cast<Scalar*>(element_type.clone().release()))), shape_(shape),
      strides_(strides), offset_(offset) {};

Tensor::Tensor(
    StorageType storage_type,
    size_t alignment,
    const std::string& initializer,
    const Scalar& element_type,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& strides,
    const symbolic::Expression& offset
)
    : IType(storage_type, alignment, initializer),
      element_type_(std::unique_ptr<Scalar>(static_cast<Scalar*>(element_type.clone().release()))), shape_(shape),
      strides_(strides), offset_(offset) {};

PrimitiveType Tensor::primitive_type() const { return this->element_type_->primitive_type(); };

bool Tensor::is_symbol() const { return false; };

const Scalar& Tensor::element_type() const { return *this->element_type_; };

const symbolic::MultiExpression& Tensor::shape() const { return this->shape_; };

const symbolic::MultiExpression& Tensor::strides() const { return this->strides_; };

const symbolic::Expression& Tensor::offset() const { return this->offset_; };

symbolic::Expression Tensor::total_elements() const {
    symbolic::Expression total = symbolic::one();
    for (const auto& dim : this->shape_) {
        total = symbolic::mul(total, dim);
    }
    return total;
};

bool Tensor::is_scalar() const { return this->shape_.empty(); }

TypeID Tensor::type_id() const { return TypeID::Tensor; };

bool Tensor::operator==(const IType& other) const {
    if (!dynamic_cast<const Tensor*>(&other)) {
        return false;
    }
    const auto& tensor_type = static_cast<const Tensor&>(other);

    if (!(*this->element_type_ == *tensor_type.element_type_)) {
        return false;
    }
    if (!symbolic::eq(this->offset_, tensor_type.offset_)) {
        return false;
    }

    if (this->shape_.size() != tensor_type.shape_.size()) {
        return false;
    }
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        if (!symbolic::eq(this->shape_.at(i), tensor_type.shape_.at(i))) {
            return false;
        }
    }

    if (this->strides_.size() != tensor_type.strides_.size()) {
        return false;
    }
    for (size_t i = 0; i < this->strides_.size(); ++i) {
        if (!symbolic::eq(this->strides_.at(i), tensor_type.strides_.at(i))) {
            return false;
        }
    }

    return true;
};

std::unique_ptr<IType> Tensor::clone() const {
    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        this->shape_,
        this->strides_,
        this->offset_
    );
};

std::string Tensor::print() const {
    std::string result = "Tensor(" + this->element_type_->print() + ", shape=[";
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        result += this->shape_.at(i)->__str__();
        if (i < this->shape_.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    result += ", strides=[";
    for (size_t i = 0; i < this->strides_.size(); ++i) {
        result += this->strides_.at(i)->__str__();
        if (i < this->strides_.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    if (!symbolic::eq(this->offset_, symbolic::zero())) {
        result += ", offset=" + this->offset_->__str__();
    }
    result += ")";
    return result;
};

std::unique_ptr<Tensor> Tensor::newaxis(size_t axis) const {
    if (axis > this->shape_.size()) {
        throw std::out_of_range("axis out of range for newaxis");
    }

    symbolic::MultiExpression new_shape = this->shape_;
    symbolic::MultiExpression new_strides = this->strides_;

    new_shape.insert(new_shape.begin() + axis, SymEngine::integer(1));
    new_strides.insert(new_strides.begin() + axis, SymEngine::integer(0));

    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        new_shape,
        new_strides,
        this->offset_
    );
}

std::unique_ptr<Tensor> Tensor::flip(size_t axis) const {
    if (axis >= this->shape_.size()) {
        throw std::out_of_range("axis out of range for flip");
    }

    symbolic::MultiExpression new_strides = this->strides_;

    // Negate the stride for the specified axis
    new_strides[axis] = SymEngine::neg(this->strides_[axis]);

    // Compute new offset: offset += stride * (shape - 1)
    auto shape_minus_one = SymEngine::sub(this->shape_[axis], SymEngine::integer(1));
    auto offset_adjustment = SymEngine::mul(this->strides_[axis], shape_minus_one);

    symbolic::Expression new_offset = SymEngine::add(this->offset_, offset_adjustment);

    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        this->shape_,
        new_strides,
        new_offset
    );
}

std::unique_ptr<Tensor> Tensor::unsqueeze(size_t axis) const { return this->newaxis(axis); }

std::unique_ptr<Tensor> Tensor::squeeze(size_t axis) const {
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

    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        new_shape,
        new_strides,
        this->offset_
    );
}

std::unique_ptr<Tensor> Tensor::squeeze() const {
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

    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        new_shape,
        new_strides,
        this->offset_
    );
}

std::unique_ptr<Tensor> Tensor::reshape(const symbolic::MultiExpression& new_shape) const {
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
    symbolic::MultiExpression new_strides = strides_from_shape(new_shape);

    return std::make_unique<Tensor>(
        this->storage_type(),
        this->alignment(),
        this->initializer(),
        *this->element_type_,
        new_shape,
        new_strides,
        this->offset_
    );
}

} // namespace types
} // namespace sdfg
