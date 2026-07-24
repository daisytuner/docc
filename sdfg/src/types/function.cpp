#include "sdfg/types/function.h"

namespace sdfg {
namespace types {

Function::Function(const IType& return_type, bool is_var_arg)
    : return_type_(return_type.clone()), is_var_arg_(is_var_arg) {};

Function::Function(
    StorageType storage_type, size_t alignment, const std::string& initializer, const IType& return_type, bool is_var_arg
)
    : IType(storage_type, alignment, initializer), return_type_(return_type.clone()), is_var_arg_(is_var_arg) {};

PrimitiveType Function::primitive_type() const { return PrimitiveType::Void; }

bool Function::is_symbol() const { return false; }

size_t Function::num_params() const { return this->params_.size(); }

const IType& Function::param_type(symbolic::Integer index) const { return *this->params_[index->as_int()]; }

void Function::add_param(const IType& param) { this->params_.push_back(param.clone()); }

const IType& Function::return_type() const { return *this->return_type_; }

TypeID Function::type_id() const { return TypeID::Function; };

bool Function::is_var_arg() const { return this->is_var_arg_; }

bool Function::operator==(const IType& other) const {
    auto other_function = dynamic_cast<const Function*>(&other);
    if (other_function == nullptr) {
        return false;
    }

    if (this->is_var_arg_ != other_function->is_var_arg_) {
        return false;
    }

    if (*this->return_type_ == *other_function->return_type_) {
        // Do nothing
    } else {
        return false;
    }

    if (this->params_.size() != other_function->params_.size()) {
        return false;
    }

    for (size_t i = 0; i < this->params_.size(); i++) {
        if (*this->params_[i] == *other_function->params_[i]) {
            continue;
        } else {
            return false;
        }
    }

    return true;
}

std::unique_ptr<IType> Function::clone() const {
    auto new_function = std::make_unique<
        Function>(this->storage_type(), this->alignment(), this->initializer(), *this->return_type_, this->is_var_arg_);
    for (const auto& param : this->params_) {
        new_function->add_param(*param);
    }
    return new_function;
}

std::string Function::print() const {
    std::string params = "";
    for (size_t i = 0; i < this->params_.size(); i++) {
        params += this->params_[i]->print();
        if (i != this->params_.size() - 1) {
            params += ", ";
        }
    }
    if (this->is_var_arg_) {
        params += ", ...)";
    } else {
        params += ")";
    }
    return "Function(" + this->return_type_->print() + ", " + params + ")";
};

void Function::replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (size_t i = 0; i < this->num_params(); i++) {
        this->params_[i]->replace_symbols(old_expression, new_expression);
    }
    this->return_type_->replace_symbols(old_expression, new_expression);
}

void Function::replace_symbols(const symbolic::ExpressionMapping& replacements) {
    for (size_t i = 0; i < this->num_params(); i++) {
        this->params_[i]->replace_symbols(replacements);
    }
    this->return_type_->replace_symbols(replacements);
}

} // namespace types
} // namespace sdfg
