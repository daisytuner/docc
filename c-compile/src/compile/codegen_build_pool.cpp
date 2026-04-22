#include "docc/compile/codegen_build_pool.h"

namespace docc::compile {

CodegenCompiler::CodegenCompiler(std::unordered_map<std::string, std::unique_ptr<CodegenCompiler>>&& redirects)
    : redirects_(std::move(redirects)) {}

std::unique_ptr<CompileState> CodegenCompiler::create_compile(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippet* snippet,
    std::function<void(std::ostream&)> generator
) {
    if (snippet) {
        auto& ext = snippet->extension();
        auto it = redirects_.find(ext);
        if (it != redirects_.end()) {
            return it->second->do_create_compile(sdfg, snippet, generator);
        }
    }

    return do_create_compile(sdfg, snippet, generator);
}

std::unique_ptr<CompileState> NoopCompiler::do_create_compile(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippet* snippet,
    std::function<void(std::ostream&)> generator
) {
    return nullptr;
}

CodegenBuildPool::CodegenBuildPool(int num_threads) : num_threads_(num_threads) {}

void CodegenBuildPool::add_compile_state(std::unique_ptr<CompileState> state) {
    auto* ptr = state.get();
    {
        std::lock_guard lock(mutex_);

        srcs_.push_back(std::move(state));
        ++outstanding_compiles_;
    }

    if (num_threads_ == 1) {
        ptr->codegen();
        ptr->compile();
        --outstanding_compiles_;
    } else {
        throw std::runtime_error("parallel build not yet implemented");
    }
}

void CodegenBuildPool::await_compiles_finished() {
    int outstanding = outstanding_compiles_.load();
    while (outstanding > 0) {
        outstanding_compiles_.wait(outstanding);
        outstanding = outstanding_compiles_.load();
    }
}

void CodegenBuildPool::for_each_src(std::function<void(CompileState&)> fn) {
    std::lock_guard lock(mutex_);

    for (auto& src : srcs_) {
        fn(*src);
    }
}

} // namespace docc::compile
