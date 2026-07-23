#include "sdfg/types/pointer.h"

namespace sdfg {
namespace types {

Pointer::Pointer() : pointee_type_(std::nullopt) {};

Pointer::Pointer(const IType& pointee_type) : pointee_type_(pointee_type.clone()) {};

Pointer::Pointer(StorageType storage_type, size_t alignment, const std::string& initializer)
    : IType(storage_type, alignment, initializer), pointee_type_(std::nullopt) {};

Pointer::Pointer(StorageType storage_type, size_t alignment, const std::string& initializer, const IType& pointee_type)
    : IType(storage_type, alignment, initializer), pointee_type_(pointee_type.clone()) {};

bool Pointer::has_pointee_type() const { return this->pointee_type_.has_value(); };

std::unique_ptr<IType> Pointer::clone() const {
    if (this->has_pointee_type()) {
        return std::make_unique<
            Pointer>(this->storage_type(), this->alignment(), this->initializer(), *this->pointee_type_.value());
    } else {
        return std::make_unique<Pointer>(this->storage_type(), this->alignment(), this->initializer());
    }
};

TypeID Pointer::type_id() const { return TypeID::Pointer; };

PrimitiveType Pointer::primitive_type() const {
    if (this->has_pointee_type()) {
        return this->pointee_type_.value()->primitive_type();
    } else {
        return PrimitiveType::Void;
    }
};

bool Pointer::is_symbol() const { return true; };

const IType& Pointer::pointee_type() const { return *this->pointee_type_.value(); };

bool Pointer::operator==(const IType& other) const {
    if (auto pointer_type = dynamic_cast<const Pointer*>(&other)) {
        if (this->has_pointee_type() != pointer_type->has_pointee_type()) {
            return false;
        }
        if (this->has_pointee_type()) {
            return *(this->pointee_type_.value()) == *pointer_type->pointee_type_.value();
        } else {
            return true;
        }
    } else {
        return false;
    }
};

std::string Pointer::print() const {
    if (this->has_pointee_type()) {
        return "Pointer(" + this->pointee_type_.value()->print() + ")";
    } else {
        return "Pointer()";
    }
};

void Pointer::replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (this->has_pointee_type()) {
        this->pointee_type_->get()->replace_symbols(old_expression, new_expression);
    }
}

void Pointer::replace_symbols(const symbolic::ExpressionMapping& replacements) {
    if (this->has_pointee_type()) {
        this->pointee_type_->get()->replace_symbols(replacements);
    }
}

} // namespace types
} // namespace sdfg
