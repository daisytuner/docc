#include "sdfg/function.h"
#include <cstddef>

#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

const std::unique_ptr<types::Scalar> Function::NVPTX_SYMBOL_TYPE =
    std::make_unique<types::Scalar>(types::PrimitiveType::UInt32);
const std::unique_ptr<types::Pointer> Function::CONST_POINTER_TYPE =
    std::make_unique<types::Pointer>(types::Scalar(types::PrimitiveType::Void));

Function::Function(const std::string& name, FunctionType type, const types::IType& return_type)
    : element_counter_(0), name_(name), type_(type), return_type_(return_type.clone()) {
    if (this->type_ == FunctionType_NV_GLOBAL) {
        this->assumptions_[symbolic::threadIdx_x()] =
            symbolic::Assumption::create(symbolic::threadIdx_x(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::threadIdx_y()] =
            symbolic::Assumption::create(symbolic::threadIdx_y(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::threadIdx_z()] =
            symbolic::Assumption::create(symbolic::threadIdx_z(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockIdx_x()] =
            symbolic::Assumption::create(symbolic::blockIdx_x(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockIdx_y()] =
            symbolic::Assumption::create(symbolic::blockIdx_y(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockIdx_z()] =
            symbolic::Assumption::create(symbolic::blockIdx_z(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockDim_x()] =
            symbolic::Assumption::create(symbolic::blockDim_x(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockDim_y()] =
            symbolic::Assumption::create(symbolic::blockDim_y(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::blockDim_z()] =
            symbolic::Assumption::create(symbolic::blockDim_z(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::gridDim_x()] =
            symbolic::Assumption::create(symbolic::gridDim_x(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::gridDim_y()] =
            symbolic::Assumption::create(symbolic::gridDim_y(), *NVPTX_SYMBOL_TYPE);
        this->assumptions_[symbolic::gridDim_z()] =
            symbolic::Assumption::create(symbolic::gridDim_z(), *NVPTX_SYMBOL_TYPE);
    }
};

Function::Function(const std::string& name, FunctionType type)
    : Function(name, type, types::Scalar(types::PrimitiveType::Void)) {}

const std::string& Function::name() const { return this->name_; };

void Function::name(const std::string& name) { this->name_ = name; };

FunctionType Function::type() const { return this->type_; };

const types::IType& Function::return_type() const { return *this->return_type_; };

size_t Function::element_counter() const { return this->element_counter_; };

void Function::validate() const {
    // Function type
    if (this->type_ != FunctionType_CPU && this->type_ != FunctionType_NV_GLOBAL) {
        throw InvalidSDFGException("Function type must be CPU or NV_GLOBAL");
    }
};

bool Function::exists(const std::string& name) const {
    return this->containers_.find(name) != this->containers_.end();
};

const types::IType& Function::type(const std::string& name) const {
    if (symbolic::is_nullptr(symbolic::symbol(name))) {
        return this->opaque_pointer_;
    }
    auto entry = this->containers_.find(name);
    if (entry == this->containers_.end()) {
        throw InvalidSDFGException("Type: Container " + name + " not found");
    }
    return *entry->second;
};

const types::StructureDefinition& Function::structure(const std::string& name) const {
    auto entry = this->structures_.find(name);
    if (entry == this->structures_.end()) {
        throw InvalidSDFGException("Structure: " + name + " not found");
    }
    return *entry->second;
};

const std::vector<std::string>& Function::arguments() const { return this->arguments_; };

const std::vector<std::string>& Function::externals() const { return this->externals_; };

LinkageType Function::linkage_type(const std::string& name) const {
    auto entry = this->externals_linkage_types_.find(name);
    if (entry == this->externals_linkage_types_.end()) {
        throw InvalidSDFGException("Linkage type: " + name + " not found");
    }
    return entry->second;
};

bool Function::is_argument(const std::string& name) const {
    return std::find(this->arguments_.begin(), this->arguments_.end(), name) != this->arguments_.end();
};

bool Function::is_external(const std::string& name) const {
    return std::find(this->externals_.begin(), this->externals_.end(), name) != this->externals_.end();
};

bool Function::is_transient(const std::string& name) const {
    return !this->is_argument(name) && !this->is_external(name);
};

symbolic::SymbolSet Function::parameters() const {
    symbolic::SymbolSet params;
    for (auto& arg : this->arguments_) {
        auto& arg_type = this->type(arg);
        if (auto scalar_type = dynamic_cast<const types::Scalar*>(&arg_type)) {
            if (scalar_type->is_symbol()) {
                params.insert(symbolic::symbol(arg));
            }
        }
    }
    for (auto& ext : this->externals_) {
        auto& ext_type = this->type(ext);
        if (auto scalar_type = dynamic_cast<const types::Scalar*>(&ext_type)) {
            if (scalar_type->is_symbol()) {
                params.insert(symbolic::symbol(ext));
            }
        }
    }
    return params;
};

bool Function::has_assumption(const symbolic::Symbol symbol) const {
    return this->assumptions_.find(symbol) != this->assumptions_.end();
};

const symbolic::Assumption& Function::assumption(const symbolic::Symbol symbol) const {
    auto entry = this->assumptions_.find(symbol);
    if (entry == this->assumptions_.end()) {
        throw InvalidSDFGException("Assumption does not exist in SDFG");
    }
    return entry->second;
};

symbolic::Assumption& Function::assumption(const symbolic::Symbol symbol) {
    auto entry = this->assumptions_.find(symbol);
    if (entry == this->assumptions_.end()) {
        throw InvalidSDFGException("Assumption does not exist in SDFG");
    }
    return entry->second;
};

const symbolic::Assumptions& Function::assumptions() const { return this->assumptions_; };

void Function::add_metadata(const std::string& key, const std::string& value) { this->metadata_[key] = value; };

void Function::remove_metadata(const std::string& key) { this->metadata_.erase(key); };

const std::string& Function::metadata(const std::string& key) const { return this->metadata_.at(key); }

const std::string* Function::metadata_if_exists(const std::string& key) const {
    auto entry = this->metadata_.find(key);
    if (entry == this->metadata_.end()) {
        return nullptr;
    }
    return &entry->second;
};

const std::unordered_map<std::string, std::string>& Function::metadata() const { return this->metadata_; };

} // namespace sdfg
