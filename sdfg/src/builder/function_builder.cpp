#include "sdfg/builder/function_builder.h"

namespace sdfg {
namespace builder {

void check_name(const std::string& name) {
    if (name.find(".") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a dot");
    } else if (name.find(" ") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a space");
    } else if (name.find("(") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a parenthesis");
    } else if (name.find(")") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a parenthesis");
    } else if (name.find("[") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a bracket");
    } else if (name.find("]") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a bracket");
    } else if (name.find("*") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a star");
    } else if (name.find("&") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a ampersand");
    } else if (name.find("!") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a bang");
    } else if (name.find("~") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a tilde");
    } else if (name.find("`") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a backtick");
    } else if (name.find("\"") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a quote");
    } else if (name.find("'") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a single quote");
    } else if (name.find(";") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a semicolon");
    } else if (name.find(":") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a colon");
    } else if (name.find(",") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a comma");
    } else if (name.find("=") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a equal sign");
    } else if (name.find("+") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a plus sign");
    } else if (name.find("-") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a minus sign");
    } else if (name.find("/") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a slash");
    } else if (name.find("%") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a percent sign");
    } else if (name.find("^") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a caret");
    } else if (name.find("|") != std::string::npos) {
        throw InvalidSDFGException("Container name " + name + " contains a pipe");
    }
};

size_t FunctionBuilder::new_element_id() const { return ++this->function().element_counter_; };

void FunctionBuilder::set_element_counter(size_t element_counter) {
    this->function().element_counter_ = element_counter;
};

void FunctionBuilder::set_return_type(const types::IType& type) const { this->function().return_type_ = type.clone(); };

const types::IType& FunctionBuilder::
    add_container(const std::string& name, const types::IType& type, bool is_argument, bool is_external) const {
    if (type.type_id() == types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "Tensor types are virtual types, add a pointer instead and use the tensor type on memlets"
        );
    }

    if (is_argument && is_external) {
        throw InvalidSDFGException("Container " + name + " cannot be both an argument and an external");
    }
    // Legal name
    check_name(name);

    auto res = this->function().containers_.insert({name, type.clone()});
    if (!res.second) {
        throw InvalidSDFGException("Container " + name + " already exists");
    }

    if (is_argument) {
        this->function().arguments_.push_back(name);
    }
    if (is_external) {
        this->function().externals_.push_back(name);
        this->function().externals_linkage_types_[name] = LinkageType_External;
    }
    if (type.is_symbol()) {
        auto sym = symbolic::symbol(name);
        this->function().assumptions_.insert({sym, symbolic::Assumption::create(sym, type)});
    }

    return *(*res.first).second;
};

const types::IType& FunctionBuilder::
    add_external(const std::string& name, const types::IType& type, LinkageType linkage_type) const {
    check_name(name);

    auto res = this->function().containers_.insert({name, type.clone()});
    if (!res.second) {
        throw InvalidSDFGException("Container " + name + " already exists");
    }

    this->function().externals_.push_back(name);
    this->function().externals_linkage_types_[name] = linkage_type;

    return *(*res.first).second;
};

void FunctionBuilder::remove_container(const std::string& name) const {
    auto& function = this->function();
    if (!function.is_transient(name)) {
        throw InvalidSDFGException("Container " + name + " is not transient");
    }
    if (this->function().containers_.find(name) == this->function().containers_.end()) {
        throw InvalidSDFGException("Container " + name + " does not exist");
    }

    auto& type = function.containers_[name];
    if (type->is_symbol() && dynamic_cast<const types::Scalar*>(type.get())) {
        function.assumptions_.erase(symbolic::symbol(name));
    }

    function.containers_.erase(name);
};

void FunctionBuilder::rename_container(const std::string& old_name, const std::string& new_name) const {
    auto& function = this->function();
    if (!function.exists(old_name)) {
        throw InvalidSDFGException("Container " + old_name + " does not exist");
    }

    // Move type
    function.containers_[new_name] = std::move(function.containers_[old_name]);
    function.containers_.erase(old_name);

    // Handling of argument
    if (function.is_argument(old_name)) {
        auto it = std::find(function.arguments_.begin(), function.arguments_.end(), old_name);
        assert(it != function.arguments_.end());
        *it = new_name;
    }
    // Handling of external
    if (function.is_external(old_name)) {
        auto it = std::find(function.externals_.begin(), function.externals_.end(), old_name);
        assert(it != function.externals_.end());
        *it = new_name;
        function.externals_linkage_types_[new_name] = function.externals_linkage_types_[old_name];
        function.externals_linkage_types_.erase(old_name);
    }
    // Handling of assumption
    if (function.assumptions().find(symbolic::symbol(old_name)) != function.assumptions().end()) {
        auto assumption = function.assumption(symbolic::symbol(old_name));

        symbolic::Assumption new_assumption(symbolic::symbol(new_name));
        new_assumption.tight_lower_bound(assumption.tight_lower_bound());
        new_assumption.tight_upper_bound(assumption.tight_upper_bound());
        for (auto& lb : assumption.lower_bounds()) {
            new_assumption.add_lower_bound(lb);
        }
        for (auto& ub : assumption.upper_bounds()) {
            new_assumption.add_upper_bound(ub);
        }

        new_assumption.constant(assumption.constant());
        new_assumption.map(assumption.map());

        function.assumptions_.erase(symbolic::symbol(old_name));
        function.assumptions_.insert({new_assumption.symbol(), new_assumption});
    }
};

void FunctionBuilder::change_type(const std::string& name, const types::IType& type) const {
    auto& function = this->function();
    if (function.containers_.find(name) == function.containers_.end()) {
        throw InvalidSDFGException("Container " + name + " does not exist");
    }

    function.containers_[name] = type.clone();
};

types::StructureDefinition& FunctionBuilder::add_structure(const std::string& name, bool is_packed) const {
    if (this->function().structures_.find(name) != this->function().structures_.end()) {
        throw InvalidSDFGException("Structure " + name + " already exists");
    }

    auto res = this->function().structures_.insert({name, std::make_unique<types::StructureDefinition>(name, is_packed)}
    );
    if (!res.second) {
        throw InvalidSDFGException("Structure " + name + " already exists");
    }

    return *(*res.first).second;
};

std::string FunctionBuilder::find_new_name(std::string prefix) const {
    size_t i = 0;
    std::string new_name = prefix + std::to_string(i);
    while (this->function().exists(new_name)) {
        i++;
        new_name = prefix + std::to_string(i);
    }
    return new_name;
};

void FunctionBuilder::update_tasklet(data_flow::Tasklet& tasklet, const data_flow::TaskletCode code) {
    tasklet.code_ = code;
}

std::unique_ptr<types::Structure> FunctionBuilder::
    create_vector_type(const types::Scalar& element_type, size_t vector_size) {
    std::string struct_name = "__daisy_vec_" + std::to_string(element_type.primitive_type()) + "_" +
                              std::to_string(vector_size);
    auto defined_structures = this->function().structures();
    if (std::find(defined_structures.begin(), defined_structures.end(), struct_name) != defined_structures.end()) {
        return std::make_unique<types::Structure>(struct_name);
    }

    auto& struct_def = this->add_structure(struct_name, true);
    for (size_t i = 0; i < vector_size; i++) {
        struct_def.add_member(element_type);
    }

    return std::make_unique<types::Structure>(struct_name);
}

} // namespace builder
} // namespace sdfg
