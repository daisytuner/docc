#include "sdfg/codegen/language_extensions/c_language_extension.h"

#include <cstddef>
#include <string>

#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

std::string CLanguageExtension::primitive_type(const types::PrimitiveType prim_type) {
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
            return "_Float16";
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

std::string CLanguageExtension::
    declaration(const std::string& name, const types::IType& type, bool use_initializer, bool use_alignment) {
    std::stringstream val;

    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        val << primitive_type(scalar_type->primitive_type());
        val << " ";
        val << name;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
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
            // ISO C forbids empty parameter lists before ...
            if (function_type->num_params() > 0) {
                params << ", ";
                params << "...";
            }
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

std::string CLanguageExtension::type_cast(const std::string& name, const types::IType& type) {
    std::stringstream val;

    val << "(";
    val << declaration("", type);
    val << ") ";
    val << name;

    return val.str();
};

std::string CLanguageExtension::subset(const types::IType& type, const data_flow::Subset& sub) {
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

std::string CLanguageExtension::expression(const symbolic::Expression expr) {
    CSymbolicPrinter printer(this->function_, this->external_prefix_);
    return printer.apply(expr);
};

std::string CLanguageExtension::access_node(const data_flow::AccessNode& node) {
    if (dynamic_cast<const data_flow::ConstantNode*>(&node)) {
        std::string name = node.data();
        if (symbolic::is_nullptr(symbolic::symbol(name))) {
            return "NULL";
        }
        return name;
    } else {
        std::string name = node.data();
        if (this->function_.is_external(name)) {
            return "(&" + this->external_prefix_ + name + ")";
        }
        return name;
    }
};

std::string CLanguageExtension::tasklet(const data_flow::Tasklet& tasklet) {
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
            return "isnan(" + tasklet.inputs().at(0) + ") && isnan(" + tasklet.inputs().at(1) + ")";
        case data_flow::TaskletCode::fp_ueq:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " == " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_une:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " != " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ugt:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " > " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_uge:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " >= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ult:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " < " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_ule:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")" + " || " +
                   tasklet.inputs().at(0) + " <= " + tasklet.inputs().at(1);
        case data_flow::TaskletCode::fp_uno:
            return "isnan(" + tasklet.inputs().at(0) + ") || isnan(" + tasklet.inputs().at(1) + ")";
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

std::string CLanguageExtension::zero(const types::PrimitiveType prim_type) {
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
            return "0u";
        case types::Half:
            throw InvalidSDFGException("Currently unsupported");
        case types::BFloat:
            throw InvalidSDFGException("Currently unsupported");
        case types::Float:
            return "0.0f";
        case types::Double:
            return "0.0";
        case types::X86_FP80:
            return "0.0l";
        case types::FP128:
            throw InvalidSDFGException("Currently unsupported");
        case types::PPC_FP128:
            throw InvalidSDFGException("Currently unsupported");
    }
}

void CSymbolicPrinter::bvisit(const SymEngine::Infty& x) {
    if (x.is_negative_infinity())
        str_ = "-INFINITY";
    else if (x.is_positive_infinity())
        str_ = "INFINITY";
};

void CSymbolicPrinter::bvisit(const SymEngine::BooleanAtom& x) { str_ = x.get_val() ? "true" : "false"; };

void CSymbolicPrinter::bvisit(const SymEngine::Symbol& x) {
    if (symbolic::is_nullptr(symbolic::symbol(x.get_name()))) {
        str_ = "((uintptr_t) NULL)";
        return;
    }

    std::string name = x.get_name();

    if (this->function_.is_external(name)) {
        name = "((uintptr_t) (&" + this->external_prefix_ + name + "))";
    } else if (this->function_.exists(name) && this->function_.type(name).type_id() == types::TypeID::Pointer) {
        name = "((uintptr_t) " + name + ")";
    }

    str_ = name;
};

void CSymbolicPrinter::bvisit(const SymEngine::And& x) {
    std::ostringstream s;
    auto container = x.get_container();
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " && " << apply(*it);
    }
    str_ = parenthesize(s.str());
};

void CSymbolicPrinter::bvisit(const SymEngine::Or& x) {
    std::ostringstream s;
    auto container = x.get_container();
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " || " << apply(*it);
    }
    str_ = parenthesize(s.str());
};

void CSymbolicPrinter::bvisit(const SymEngine::Not& x) {
    str_ = "!" + apply(x.get_arg());
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Equality& x) {
    str_ = apply(x.get_args()[0]) + " == " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Unequality& x) {
    str_ = apply(x.get_args()[0]) + " != " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Min& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        if (this->use_rtl_functions_) {
            s << "__daisy_min(";
        } else {
            s << "min(";
        }
        s << apply(*container.begin());

        // Recursively apply __daisy_min to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submin = SymEngine::min(subargs);
        s << ", " << apply(submin);

        s << ")";
    }

    str_ = s.str();
};

void CSymbolicPrinter::bvisit(const SymEngine::Max& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        if (this->use_rtl_functions_) {
            s << "__daisy_max(";
        } else {
            s << "max(";
        }
        s << apply(*container.begin());

        // Recursively apply __daisy_max to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submax = SymEngine::max(subargs);
        s << ", " << apply(submax);

        s << ")";
    }

    str_ = s.str();
};

void CSymbolicPrinter::bvisit(const SymEngine::FunctionSymbol& x) {
    if (x.get_name() == "idiv") {
        str_ = "((" + apply(x.get_args()[0]) + ") / (" + apply(x.get_args()[1]) + "))";
    } else if (x.get_name() == "iabs") {
        str_ = "((" + apply(x.get_args()[0]) + ") < 0 ? -(" + apply(x.get_args()[0]) + ") : (" +
               apply(x.get_args()[0]) + "))";
    } else if (x.get_name() == "zext_i64") {
        str_ = "((long long) ((unsigned long long) (" + apply(x.get_args()[0]) + ")))";
    } else if (x.get_name() == "trunc_i32") {
        str_ = "((int) ((unsigned int) ((unsigned long long) (" + apply(x.get_args()[0]) + "))))";
    } else if (x.get_name() == "sizeof") {
        auto& so = dynamic_cast<const symbolic::SizeOfTypeFunction&>(x);
        auto& type = so.get_type();
        CLanguageExtension lang(this->function_, this->external_prefix_);
        str_ = "sizeof(" + lang.declaration("", type) + ")";
    } else if (x.get_name() == "malloc_usable_size") {
        str_ = "malloc_usable_size(" +
               SymEngine::rcp_static_cast<const SymEngine::Symbol>(x.get_args()[0])->get_name() + ")";
    } else {
        throw std::runtime_error("Unsupported function symbol: " + x.get_name());
    }
};

void CSymbolicPrinter::_print_pow(
    std::ostringstream& o,
    const SymEngine::RCP<const SymEngine::Basic>& a,
    const SymEngine::RCP<const SymEngine::Basic>& b
) {
    if (SymEngine::eq(*a, *SymEngine::E)) {
        o << "exp(" << apply(b) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 2))) {
        o << "sqrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 3))) {
        o << "cbrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::integer(2))) {
        o << "((" + apply(a) + ") * (" + apply(a) + "))";
    } else {
        o << "__daisy_sym_pow(" << apply(a) << ", " << apply(b) << ")";
    }
};

} // namespace codegen
} // namespace sdfg
