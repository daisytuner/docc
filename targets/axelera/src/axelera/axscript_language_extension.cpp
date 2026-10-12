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

std::string AxScriptLanguageExtension::
    dpu_operation(data_flow::TaskletCode code, const std::vector<std::string>& operands) const {
    switch (code) {
        case data_flow::TaskletCode::assign:
            return operands.at(0);
        case data_flow::TaskletCode::fp_add:
            return operands.at(0) + " + " + operands.at(1);
        case data_flow::TaskletCode::fp_sub:
            return operands.at(0) + " - " + operands.at(1);
        case data_flow::TaskletCode::fp_mul:
            return operands.at(0) + " * " + operands.at(1);
        default:
            throw std::runtime_error(
                "AxScript codegen: tasklet code " + std::to_string(static_cast<int>(code)) +
                " on the DPU is not supported"
            );
    }
}

} // namespace axelera
} // namespace sdfg
