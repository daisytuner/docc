#pragma once

#include <sdfg/codegen/language_extensions/python_language_extension.h>

#include "docc/target/axelera/datapath.h"

namespace sdfg {
namespace axelera {

/// AxScript (Axelera's TVMScript dialect, imported as `A`): Python expressions plus datapath types and DPU operations.
class AxScriptLanguageExtension : public codegen::PythonLanguageExtension {
private:
    size_t datapaths_ = 0;
    std::vector<double> constants_;

public:
    using PythonLanguageExtension::PythonLanguageExtension;

    const std::string language() const override {
        return "AxScript";
    }

    /// AXELERA loops have their own dispatcher, so no schedule falls back to a sequential loop.
    std::string loop_schedule(const structured_control_flow::ScheduleType& schedule) const override {
        return schedule.value();
    }

    /// dtype of a PWORD stream of @p prim elements, e.g. "float32x64".
    std::string vector_type(types::PrimitiveType prim) const;

    /// dtype of the DPU registers, which compute at reduced precision.
    static std::string dpu_type();

    /// DPU expression applying @p op to register expressions @p operands.
    std::string dpu_operation(DpuOp op, const std::vector<std::string>& operands) const;

    /// DPU expression reducing the lanes of register @p value with @p op, broadcast back to all lanes.
    std::string lane_reduction(DpuOp op, const std::string& value) const;

    /// Distinguishes the stream and register names of each datapath in a module.
    size_t next_datapath() {
        return datapaths_++;
    }

    /// Name of the module's constant buffer holding @p value in every lane, registering it on first use.
    std::string constant_buffer(double value);

    /// Values of the registered constant buffers; buffer k is named `constant_name(k)`.
    const std::vector<double>& constants() const {
        return constants_;
    }

    static std::string constant_name(size_t k);

    /// Python literal for @p value, spelling non-finite values as `float("inf")` etc.
    static std::string float_literal(double value);
};

} // namespace axelera
} // namespace sdfg
