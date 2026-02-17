#include "sdfg/codegen/language_extensions/hip_language_extension.h"

#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"

namespace sdfg {
namespace codegen {

std::string HIPLanguageExtension::primitive_type(const types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Void:
            return "void";
        case types::PrimitiveType::Bool:
            return "bool";
        case types::PrimitiveType::Int8:
            return "signed char";
        case types::PrimitiveType::Int16:
            return "short";
        case types::PrimitiveType::Int32:
            return "int";
        case types::PrimitiveType::Int64:
            return "long long";
        case types::PrimitiveType::Int128:
            return "__int128";
        case types::PrimitiveType::UInt8:
            return "char";
        case types::PrimitiveType::UInt16:
            return "unsigned short";
        case types::PrimitiveType::UInt32:
            return "unsigned int";
        case types::PrimitiveType::UInt64:
            return "unsigned long long";
        case types::PrimitiveType::UInt128:
            return "unsigned __int128";
        case types::PrimitiveType::Half:
            return "__fp16";
        case types::PrimitiveType::BFloat:
            return "__bf16";
        case types::PrimitiveType::Float:
            return "float";
        case types::PrimitiveType::Double:
            return "double";
        case types::PrimitiveType::X86_FP80:
            return "long double";
        case types::PrimitiveType::FP128:
            return "__float128";
        case types::PrimitiveType::PPC_FP128:
            return "__float128";
    }

    throw std::runtime_error("Unknown primitive type");
};

std::string HIPLanguageExtension::
    declaration(const std::string& name, const types::IType& type, bool use_initializer, bool use_alignment) {
    std::stringstream val;

    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        if (scalar_type->storage_type().is_nv_shared()) {
            val << "__shared__ ";
        } else if (scalar_type->storage_type().is_nv_constant()) {
            val << "__constant__ ";
        }
        val << primitive_type(scalar_type->primitive_type());
        val << " ";
        val << name;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        if (array_type->storage_type().is_nv_shared()) {
            val << "__shared__ ";
        }
        auto& element_type = array_type->element_type();
        val << declaration(name + "[" + this->expression(array_type->num_elements()) + "]", element_type);
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        if (pointer_type->has_pointee_type()) {
            const types::IType& pointee = pointer_type->pointee_type();

            const bool pointee_is_function_or_array = dynamic_cast<const types::Function*>(&pointee) ||
                                                      dynamic_cast<const types::Array*>(&pointee);

            // Parenthesise *only* when it is needed to bind tighter than [] or ()
            std::string decorated = pointee_is_function_or_array ? "(*" + name + ")" : "*" + name;

            val << declaration(decorated, pointee);
        } else {
            val << "void*";
            val << " " << name;
        }
    } else if (auto ref_type = dynamic_cast<const Reference*>(&type)) {
        val << declaration("&" + name, ref_type->reference_type());
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        if (structure_type->storage_type().is_nv_shared()) {
            val << "__shared__ ";
        } else if (structure_type->storage_type().is_nv_constant()) {
            val << "__constant__ ";
        }
        val << structure_type->name();
        val << " ";
        val << name;
    } else if (auto function_type = dynamic_cast<const types::Function*>(&type)) {
        std::stringstream params;
        for (size_t i = 0; i < function_type->num_params(); ++i) {
            params << declaration("", function_type->param_type(symbolic::integer(i)));
            if (i + 1 < function_type->num_params()) params << ", ";
        }
        if (function_type->is_var_arg()) {
            if (function_type->num_params() > 0) {
                params << ", ";
            }
            params << "...";
        }

        const std::string fun_name = name + "(" + params.str() + ")";
        val << declaration(fun_name, function_type->return_type());
    } else {
        throw std::runtime_error("Unknown declaration type");
    }

    if (use_alignment && type.alignment() > 0) {
        val << " __attribute__((aligned(" << type.alignment() << ")))";
    }

    if (use_initializer && !type.initializer().empty()) {
        val << " = " << type.initializer();
    }

    return val.str();
};

std::string HIPLanguageExtension::type_cast(const std::string& name, const types::IType& type) {
    std::stringstream val;

    val << "reinterpret_cast";
    val << "<";
    val << declaration("", type);
    val << ">";
    val << "(" << name << ")";

    return val.str();
};

std::string HIPLanguageExtension::subset(const types::IType& type, const data_flow::Subset& sub) {
    if (sub.empty()) {
        return "";
    }

    if (dynamic_cast<const types::Scalar*>(&type)) {
        return "";
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        std::string subset_str = "[" + this->expression(sub.at(0)) + "]";

        if (sub.size() > 1) {
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            auto& element_type = array_type->element_type();
            return subset_str + subset(element_type, element_subset);
        } else {
            return subset_str;
        }
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        std::string subset_str = "[" + this->expression(sub.at(0)) + "]";

        data_flow::Subset element_subset(sub.begin() + 1, sub.end());
        auto& pointee_type = pointer_type->pointee_type();
        return subset_str + subset(pointee_type, element_subset);
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        auto& definition = this->function_.structure(structure_type->name());

        std::string subset_str = ".member_" + this->expression(sub.at(0));
        if (sub.size() > 1) {
            auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(sub.at(0));
            auto& member_type = definition.member_type(member);
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            return subset_str + subset(member_type, element_subset);
        } else {
            return subset_str;
        }
    }

    throw std::invalid_argument("Invalid subset type");
};

std::string HIPLanguageExtension::expression(const symbolic::Expression expr) {
    CPPSymbolicPrinter printer(this->function_, this->external_prefix_);
    return printer.apply(expr);
};

std::string HIPLanguageExtension::access_node(const data_flow::AccessNode& node) {
    if (dynamic_cast<const data_flow::ConstantNode*>(&node)) {
        std::string name = node.data();
        if (symbolic::is_nullptr(symbolic::symbol(name))) {
            return "nullptr";
        }
        return name;
    } else {
        std::string name = node.data();
        if (this->function_.is_external(name)) {
            return "(&" + name + ")";
        }
        return name;
    }
};

std::string HIPLanguageExtension::tasklet(const data_flow::Tasklet& tasklet) {
    switch (tasklet.code()) {
        case data_flow::TaskletCode::assign:
            return tasklet.inputs().at(0);
        case data_flow::TaskletCode::fp_neg:
            return "-" + tasklet.inputs().at(0);
        case data_flow::TaskletCode::fp_add:
            return tasklet.inputs().at(0) + " + " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_sub:
            return tasklet.inputs().at(0) + " - " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_mul:
            return tasklet.inputs().at(0) + " * " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_div:
            return tasklet.inputs().at(0) + " / " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_rem:
            return "fmod(" + tasklet.inputs().at(0) + ", " + tasklet.inputs().at(1) + ")";
        case data_flow::TaskletCode::fp_fma:
            return tasklet.inputs().at(0) + " * " + tasklet.inputs().at(1) + " + " + tasklet.inputs().at(2);
        case data_flow::TaskletCode::fp_oeq:
            return tasklet.inputs().at(0) + " == " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_one:
            return tasklet.inputs().at(0) + " != " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ogt:
            return tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_oge:
            return tasklet.inputs().at(0) + " >= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_olt:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ole:
            return tasklet.inputs().at(0) + " <= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ord:
            return "std::isnan(" + tasklet.inputs().at(0) + ") && std::isnan(" + tasklet.inputs().at(1) + ")";
        case data_flow::TaskletCode::fp_ueq:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " == " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_une:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " != " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ugt:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_uge:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " >= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ult:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ule:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " <= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_uno:
            return "std::isnan(" + tasklet.inputs().at(0) + ") || std::isnan(" + tasklet.inputs().at(1) + ")";
        case data_flow::TaskletCode::int_add:
            return tasklet.inputs().at(0) + " + " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_sub:
            return tasklet.inputs().at(0) + " - " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_mul:
            return tasklet.inputs().at(0) + " * " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_sdiv:
            return tasklet.inputs().at(0) + " / " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_srem:
            return tasklet.inputs().at(0) + " % " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_udiv:
            return tasklet.inputs().at(0) + " / " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_urem:
            return tasklet.inputs().at(0) + " % " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_and:
            return tasklet.inputs().at(0) + " & " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_or:
            return tasklet.inputs().at(0) + " | " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_xor:
            return tasklet.inputs().at(0) + " ^ " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_shl:
            return tasklet.inputs().at(0) + " << " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_lshr:
            return tasklet.inputs().at(0) + " >> " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ashr:
            return tasklet.inputs().at(0) + " >> " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_smin:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1) + " ? " + tasklet.inputs().at(0) + " : " +
                   tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_smax:
            return tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1) + " ? " + tasklet.inputs().at(0) + " : " +
                   tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_scmp:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1) + " ? -1 : (" + tasklet.inputs().at(0) +
                   " > " + tasklet.inputs().at(1) + " ? 1 : 0)";
        case data_flow::TaskletCode::int_umin:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1) + " ? " + tasklet.inputs().at(0) + " : " +
                   tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_umax:
            return tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1) + " ? " + tasklet.inputs().at(0) + " : " +
                   tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ucmp:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1) + " ? -1 : (" + tasklet.inputs().at(0) +
                   " > " + tasklet.inputs().at(1) + " ? 1 : 0)";
        case data_flow::TaskletCode::int_abs:
            return "(" + tasklet.inputs().at(0) + " < 0 ? -" + tasklet.inputs().at(0) + " : " + tasklet.inputs().at(0) +
                   ")";
        case data_flow::TaskletCode::int_eq:
            return tasklet.inputs().at(0) + " == " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ne:
            return tasklet.inputs().at(0) + " != " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_sgt:
            return tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_sge:
            return tasklet.inputs().at(0) + " >= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_slt:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_sle:
            return tasklet.inputs().at(0) + " <= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ugt:
            return tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_uge:
            return tasklet.inputs().at(0) + " >= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ult:
            return tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::int_ule:
            return tasklet.inputs().at(0) + " <= " + tasklet.inputs().at(1);
    };
    throw std::invalid_argument("Invalid tasklet code");
};

std::string HIPLanguageExtension::zero(const types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::Void:
            throw InvalidSDFGException("No zero for void type possible");
        case types::Bool:
            return "false";
        case types::Int8:
            return "0";
        case types::Int16:
            return "0";
        case types::Int32:
            return "0";
        case types::Int64:
            return "0ll";
        case types::Int128:
            return "0";
        case types::UInt8:
            return "0u";
        case types::UInt16:
            return "0u";
        case types::UInt32:
            return "0u";
        case types::UInt64:
            return "0ull";
        case types::UInt128:
            return "0";
        case types::Half:
            return "(__fp16)0.0f";
        case types::BFloat:
            return "(__bf16)0.0f";
        case types::Float:
            return "0.0f";
        case types::Double:
            return "0.0";
        case types::X86_FP80:
            return "0.0l";
        case types::FP128:
            return "0.0";
        case types::PPC_FP128:
            return "0.0";
    }
}

} // namespace codegen
} // namespace sdfg
