#include "docc/compile/codegen_compiler.h"

namespace docc::compile {

std::unique_ptr<CompileState> NoopCompiler::create_compile(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippet* snippet,
    std::function<void(std::ostream&)> generator
) {
    return nullptr;
}

} // namespace docc::compile
