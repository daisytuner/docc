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

Tensor::Tensor(const Scalar& element_type, const math::tensor::TensorLayout& layout)
    : element_type_(std::unique_ptr<Scalar>(static_cast<Scalar*>(element_type.clone().release()))), layout_(layout) {};

Tensor::Tensor(const Scalar& element_type, const symbolic::MultiExpression& shape)
    : Tensor(element_type, math::tensor::TensorLayout(shape)) {};

Tensor::Tensor(
    const Scalar& element_type,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& strides,
    const symbolic::Expression& offset
)
    : Tensor(element_type, math::tensor::TensorLayout(shape, strides, offset)) {};

Tensor::Tensor(
    StorageType storage_type,
    size_t alignment,
    const std::string& initializer,
    const Scalar& element_type,
    const math::tensor::TensorLayout& layout
)
    : IType(storage_type, alignment, initializer),
      element_type_(std::unique_ptr<Scalar>(static_cast<Scalar*>(element_type.clone().release()))), layout_(layout) {};

Tensor::Tensor(
    StorageType storage_type,
    size_t alignment,
    const std::string& initializer,
    const Scalar& element_type,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& strides,
    const symbolic::Expression& offset
)
    : Tensor(
          std::move(storage_type),
          alignment,
          initializer,
          element_type,
          math::tensor::TensorLayout(shape, strides, offset)
      ) {};

PrimitiveType Tensor::primitive_type() const { return this->element_type_->primitive_type(); };

bool Tensor::is_symbol() const { return false; };

const Scalar& Tensor::element_type() const { return *this->element_type_; };

const math::tensor::TensorLayout& Tensor::layout() const { return this->layout_; }

const symbolic::MultiExpression& Tensor::shape() const { return this->layout_.shape(); };

const symbolic::MultiExpression& Tensor::strides() const { return this->layout_.strides(); };

const symbolic::Expression& Tensor::offset() const { return this->layout_.offset(); };

symbolic::Expression Tensor::total_elements() const { return layout_.total_elements(); };

bool Tensor::is_scalar() const { return this->layout_.shape().empty(); }

TypeID Tensor::type_id() const { return TypeID::Tensor; };

bool Tensor::operator==(const IType& other) const {
    if (!dynamic_cast<const Tensor*>(&other)) {
        return false;
    }
    const auto& tensor_type = static_cast<const Tensor&>(other);

    if (!this->element_type_->operator==(*tensor_type.element_type_)) {
        return false;
    }
    if (layout_ != tensor_type.layout_) {
        return false;
    }

    return true;
};


std::unique_ptr<IType> Tensor::clone() const {
    return std::make_unique<
        Tensor>(this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, this->layout_);
};

std::string Tensor::print() const {
    std::string result = "Tensor(" + this->element_type_->print() + ", ";
    result += layout_.toStr();
    result += ")";
    return result;
};

std::unique_ptr<Tensor> Tensor::newaxis(size_t axis) const {
    return std::make_unique<Tensor>(
        this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, *layout_.newaxis(axis)
    );
}

std::unique_ptr<Tensor> Tensor::flip(size_t axis) const {
    return std::make_unique<
        Tensor>(this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, *layout_.flip(axis));
}

std::unique_ptr<Tensor> Tensor::unsqueeze(size_t axis) const { return this->newaxis(axis); }

std::unique_ptr<Tensor> Tensor::squeeze(size_t axis) const {
    return std::make_unique<Tensor>(
        this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, *layout_.squeeze(axis)
    );
}

std::unique_ptr<Tensor> Tensor::squeeze() const {
    return std::make_unique<
        Tensor>(this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, *layout_.squeeze());
}

std::unique_ptr<Tensor> Tensor::reshape(const symbolic::MultiExpression& new_shape) const {
    return std::make_unique<Tensor>(
        this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, *layout_.reshape(new_shape)
    );
}

} // namespace types
} // namespace sdfg
