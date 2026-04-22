#pragma once
#include <functional>
#include <memory>
#include <sdfg/codegen/code_snippet_factory.h>
#include <sdfg/structured_sdfg.h>

namespace docc::compile {

class CodegenCompiler;

class CompileState {
public:
    virtual ~CompileState() = default;
    virtual bool codegen() = 0;
    virtual bool compile() = 0;
    [[nodiscard]] virtual CodegenCompiler& creator() const = 0;
};

class CodegenCompiler {
public:
    virtual ~CodegenCompiler() = default;

    virtual std::unique_ptr<CompileState> create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) = 0;
};

class NoopCompiler : public CodegenCompiler {
public:
    NoopCompiler() = default;

    std::unique_ptr<CompileState> create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) override;
};

} // namespace docc::compile
