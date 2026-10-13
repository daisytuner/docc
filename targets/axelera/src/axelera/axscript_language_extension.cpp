#include "docc/target/axelera/axscript_language_extension.h"

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

} // namespace axelera
} // namespace sdfg
