#include "docc/target/axelera/axscript_language_extension.h"

#include <bit>
#include <charconv>
#include <cmath>
#include <stdexcept>

#include "docc/target/axelera/schedule.h"

namespace sdfg {
namespace axelera {

std::string AxScriptLanguageExtension::vector_type(types::PrimitiveType prim) const {
    if (prim != types::PrimitiveType::Float) {
        throw std::runtime_error(
            std::string("AxScript codegen: streams of ") + types::primitive_type_to_string(prim) + " are not supported"
        );
    }
    return "float32x" + std::to_string(PWORD_LANES);
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
