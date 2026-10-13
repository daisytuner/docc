#include "docc/target/axelera/axscript_language_extension.h"

#include <bit>
#include <charconv>
#include <cmath>
#include <stdexcept>

#include "docc/target/axelera/schedule.h"

namespace sdfg {
namespace axelera {

namespace {

/// TIR expressions as AxScript spells them. Integer division and remainder become floor division and modulo, which
/// agree with C on the non-negative indices of tiles and pipeline stages.
class AxScriptSymbolicPrinter : public SymEngine::BaseVisitor<AxScriptSymbolicPrinter, SymEngine::CodePrinter> {
    /// `fn` applied to the operands pairwise, as the TIR intrinsics take two.
    template<typename Operands>
    std::string pairwise(const std::string& fn, const Operands& operands) {
        std::string result;
        for (auto& operand : operands) {
            result = result.empty() ? apply(operand) : fn + "(" + result + ", " + apply(operand) + ")";
        }
        return result;
    }

public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    void bvisit(const SymEngine::And& x) {
        str_ = pairwise("A.And", x.get_container());
    }

    void bvisit(const SymEngine::Or& x) {
        str_ = pairwise("A.Or", x.get_container());
    }

    void bvisit(const SymEngine::Not& x) {
        str_ = "A.Not(" + apply(x.get_arg()) + ")";
    }

    void bvisit(const SymEngine::Equality& x) {
        str_ = parenthesize(apply(x.get_args()[0]) + " == " + apply(x.get_args()[1]));
    }

    void bvisit(const SymEngine::Unequality& x) {
        str_ = parenthesize(apply(x.get_args()[0]) + " != " + apply(x.get_args()[1]));
    }

    void bvisit(const SymEngine::Min& x) {
        str_ = pairwise("A.min", x.get_args());
    }

    void bvisit(const SymEngine::Max& x) {
        str_ = pairwise("A.max", x.get_args());
    }

    void bvisit(const SymEngine::FunctionSymbol& x) {
        auto args = x.get_args();
        auto operand = [&](const symbolic::Expression& arg) {
            bool atomic = SymEngine::is_a<SymEngine::Symbol>(*arg) ||
                          (SymEngine::is_a<SymEngine::Integer>(*arg) &&
                           !SymEngine::down_cast<const SymEngine::Integer&>(*arg).is_negative());
            return atomic ? apply(arg) : "(" + apply(arg) + ")";
        };
        if (x.get_name() == "idiv") {
            str_ = parenthesize(operand(args[0]) + " // " + operand(args[1]));
        } else if (x.get_name() == "imod") {
            str_ = parenthesize(operand(args[0]) + " % " + operand(args[1]));
        } else {
            throw std::runtime_error("AxScript codegen: function symbol " + x.get_name() + " is not supported");
        }
    }
};

} // namespace

std::string AxScriptLanguageExtension::expression(const symbolic::Expression expr) {
    AxScriptSymbolicPrinter printer;
    return printer.apply(expr);
}

std::string AxScriptLanguageExtension::vector_type(types::PrimitiveType prim) const {
    switch (prim) {
        case types::PrimitiveType::Float:
            return "float32x" + std::to_string(PWORD_LANES);
        case types::PrimitiveType::Int8:
            return "int8x" + std::to_string(PWORD_LANES);
        default:
            throw std::runtime_error(
                std::string("AxScript codegen: buffers of ") + types::primitive_type_to_string(prim) +
                " are not supported"
            );
    }
}

std::string AxScriptLanguageExtension::dpu_type() {
    return "float18x" + std::to_string(PWORD_LANES);
}

std::string AxScriptLanguageExtension::dpu_operation(DpuOp op, const std::vector<std::string>& operands) const {
    switch (op) {
        case DpuOp::Copy:
            return operands.at(0);
        case DpuOp::Add:
            return operands.at(0) + " + " + operands.at(1);
        case DpuOp::Sub:
            return operands.at(0) + " - " + operands.at(1);
        case DpuOp::Mul:
            return operands.at(0) + " * " + operands.at(1);
        case DpuOp::Max:
            return "A.max(" + operands.at(0) + ", " + operands.at(1) + ")";
        case DpuOp::Min:
            return "A.min(" + operands.at(0) + ", " + operands.at(1) + ")";
        case DpuOp::Fma:
            return "A.dpu.ops.fma(" + operands.at(0) + ", " + operands.at(1) + ", " + operands.at(2) + ")";
        case DpuOp::Neg:
            return "A.dpu.ops.neg(" + operands.at(0) + ")";
        case DpuOp::Recip:
            return "A.dpu.ops.recip(" + operands.at(0) + ")";
        case DpuOp::Sqrt:
            return "A.dpu.ops.sqrt(" + operands.at(0) + ")";
        case DpuOp::Sin:
            return "A.dpu.ops.sin(" + operands.at(0) + ")";
        case DpuOp::Cos:
            return "A.dpu.ops.cos(" + operands.at(0) + ")";
        case DpuOp::Log2:
            return "A.dpu.ops.log2(" + operands.at(0) + ")";
        case DpuOp::Exp2:
            return "A.dpu.ops.exp2(" + operands.at(0) + ")";
    }
    throw std::invalid_argument("Invalid DpuOp");
}

std::string AxScriptLanguageExtension::lane_reduction(DpuOp op, const std::string& value) const {
    switch (op) {
        case DpuOp::Add:
            return "A.dpu.ops.add_reduce(" + value + ")";
        case DpuOp::Max:
            return "A.dpu.ops.max_reduce(" + value + ")";
        case DpuOp::Min:
            return "A.dpu.ops.min_reduce(" + value + ")";
        default:
            throw std::runtime_error("AxScript codegen: reducing lanes by this operation is not supported");
    }
}

std::string AxScriptLanguageExtension::constant_buffer(double value) {
    for (size_t k = 0; k < constants_.size(); ++k) {
        if (std::bit_cast<uint64_t>(constants_[k]) == std::bit_cast<uint64_t>(value)) {
            return constant_name(k);
        }
    }
    constants_.push_back(value);
    return constant_name(constants_.size() - 1);
}

std::string AxScriptLanguageExtension::constant_name(size_t k) {
    return "_docc_c" + std::to_string(k);
}

std::string AxScriptLanguageExtension::float_literal(double value) {
    if (std::isnan(value)) {
        return "float(\"nan\")";
    }
    if (std::isinf(value)) {
        return value > 0 ? "float(\"inf\")" : "float(\"-inf\")";
    }
    char buffer[32];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    std::string text(buffer, result.ptr);
    if (text.find_first_of(".e") == std::string::npos) {
        text += ".0";
    }
    return text;
}

} // namespace axelera
} // namespace sdfg
