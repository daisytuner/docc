#include "sdfg/types/array.h"

namespace sdfg {
namespace types {

Array::Array(const IType& element_type, const symbolic::Expression num_elements)
    : element_type_(element_type.clone()), num_elements_(num_elements) {};

Array::Array(
    StorageType storage_type,
    size_t alignment,
    const std::string& initializer,
    const IType& element_type,
    const symbolic::Expression num_elements
)
    : IType(storage_type, alignment, initializer), element_type_(element_type.clone()), num_elements_(num_elements) {};

PrimitiveType Array::primitive_type() const { return this->element_type_->primitive_type(); };

bool Array::is_symbol() const { return false; };

const IType& Array::element_type() const { return *this->element_type_; };

const symbolic::Expression Array::num_elements() const { return this->num_elements_; };

TypeID Array::type_id() const { return TypeID::Array; };

bool Array::operator==(const IType& other) const {
    if (auto array_type = dynamic_cast<const Array*>(&other)) {
        return symbolic::eq(this->num_elements_, array_type->num_elements_) &&
               *(this->element_type_) == *array_type->element_type_;
    } else {
        return false;
    }
};

std::unique_ptr<IType> Array::clone() const {
    return std::make_unique<
        Array>(this->storage_type(), this->alignment(), this->initializer(), *this->element_type_, this->num_elements_);
};

std::string Array::print() const { return "Array(" + this->element_type_->print() + ")"; };

void Array::replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->element_type_->replace_symbols(old_expression, new_expression);
    this->num_elements_ = symbolic::subs(this->num_elements_, old_expression, new_expression);
}

void Array::replace_symbols(const symbolic::ExpressionMapping& replacements) {
    this->element_type_->replace_symbols(replacements);
    this->num_elements_ = symbolic::subs(this->num_elements_, replacements);
}

} // namespace types
} // namespace sdfg
