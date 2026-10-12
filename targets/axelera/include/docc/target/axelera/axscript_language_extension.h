#pragma once

#include <sdfg/codegen/language_extensions/python_language_extension.h>

namespace sdfg {
namespace axelera {

/// AxScript (Axelera's TVMScript dialect, imported as `A`): Python expressions plus datapath types and DPU operations.
class AxScriptLanguageExtension : public codegen::PythonLanguageExtension {
private:
    size_t datapaths_ = 0;

public:
    using PythonLanguageExtension::PythonLanguageExtension;

    const std::string language() const override {
        return "AxScript";
    }

    /// dtype of a PWORD stream of @p prim elements, e.g. "float32x64".
    std::string vector_type(types::PrimitiveType prim) const;

    /// dtype of the DPU registers, which compute at reduced precision.
    static std::string dpu_type();

    /// DPU expression applying @p code to register expressions @p operands.
    std::string dpu_operation(data_flow::TaskletCode code, const std::vector<std::string>& operands) const;

    /// Distinguishes the stream and register names of each datapath in a module.
    size_t next_datapath() {
        return datapaths_++;
    }
};

} // namespace axelera
} // namespace sdfg
