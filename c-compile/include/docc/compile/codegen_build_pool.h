#pragma once
#include <list>
#include <memory>

#include "docc/util/docc_paths.h"
#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/structured_sdfg.h"

namespace docc::compile {

class CompileState;

class CodegenCompiler {
protected:
    std::unordered_map<std::string, std::unique_ptr<CodegenCompiler>> redirects_;

public:
    CodegenCompiler() = default;
    CodegenCompiler(std::unordered_map<std::string, std::unique_ptr<CodegenCompiler>>&& redirects);
    virtual ~CodegenCompiler() = default;

    std::unique_ptr<CompileState> create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    );

    virtual std::unique_ptr<CompileState> do_create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) = 0;
};

class NoopCompiler : public CodegenCompiler {
public:
    NoopCompiler() = default;

    std::unique_ptr<CompileState> do_create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) override;
};

template<typename T>
class CodegenCompilerBuilderBase {
protected:
    std::unordered_map<std::string, std::unique_ptr<CodegenCompiler>> redirects_;
    std::shared_ptr<util::DefaultDoccPaths> docc_paths_;

public:
    virtual ~CodegenCompilerBuilderBase() = default;


    T& redirect_snippet(const std::string& ext, std::unique_ptr<CodegenCompiler> handler) {
        redirects_[ext] = std::move(handler);
        return *dynamic_cast<T*>(this);
    }

    virtual T& set_from_paths(std::shared_ptr<util::DefaultDoccPaths> paths) {
        docc_paths_ = paths;
        return *dynamic_cast<T*>(this);
    }

    const util::DefaultDoccPaths& docc_paths() const { return *docc_paths_; }
};

class CompileExecutor {
public:
    virtual ~CompileExecutor() = default;

    virtual void add_compile_state(std::unique_ptr<CompileState> state) = 0;
    virtual void await_compiles_finished() = 0;

    virtual void for_each_src(std::function<void(CompileState&)> fn) = 0;
};

class CodegenBuildPool : public CompileExecutor {
private:
    std::vector<std::unique_ptr<CompileState>> srcs_;
    std::mutex mutex_;
    int num_threads_;
    std::atomic_int outstanding_compiles_ = 0;

public:
    CodegenBuildPool(int num_threads);

    void add_compile_state(std::unique_ptr<CompileState> state) override;
    void await_compiles_finished() override;

    void for_each_src(std::function<void(CompileState&)> fn) override;
};

class CompileState {
public:
    virtual ~CompileState() = default;
    virtual bool codegen() = 0;
    virtual bool compile() = 0;
    [[nodiscard]] virtual CodegenCompiler& creator() const = 0;
};

} // namespace docc::compile
