#include "sdfg/types/scalar.h"

namespace sdfg {
namespace types {

Scalar::Scalar(PrimitiveType primitive_type) : primitive_type_(primitive_type) {};

Scalar::Scalar(StorageType storage_type, size_t alignment, const std::string& initializer, PrimitiveType primitive_type)
    : IType(storage_type, alignment, initializer), primitive_type_(primitive_type) {};

PrimitiveType Scalar::primitive_type() const {
    return this->primitive_type_;
};

bool Scalar::is_symbol() const {
    return types::is_integer(this->primitive_type_);
};

TypeID Scalar::type_id() const {
    return TypeID::Scalar;
};

bool Scalar::operator==(const IType& other) const {
    if (auto scalar_type = dynamic_cast<const Scalar*>(&other)) {
        return this->primitive_type_ == scalar_type->primitive_type_;
    } else {
        return false;
    }
};

Scalar Scalar::as_signed() const {
    return Scalar(this->storage_type(), this->alignment(), this->initializer(), types::as_signed(this->primitive_type_));
};

Scalar Scalar::as_unsigned() const {
    return Scalar(this->storage_type(), this->alignment(), this->initializer(), types::as_unsigned(this->primitive_type_));
};

std::unique_ptr<IType> Scalar::clone() const {
    return std::make_unique<Scalar>(this->storage_type(), this->alignment(), this->initializer(), this->primitive_type_);
};

std::string Scalar::print() const {
    return "Scalar(" + std::string(primitive_type_to_string(this->primitive_type_)) + ")";
};

} // namespace types
} // namespace sdfg
