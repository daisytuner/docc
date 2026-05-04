#pragma once
#include <condition_variable>
#include <list>
#include <memory>
#include <queue>
#include <thread>

#include "docc/compile/codegen_compiler.h"
#include "docc/util/docc_paths.h"
#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/structured_sdfg.h"

namespace docc::compile {

class CompileExecutor {
public:
    virtual ~CompileExecutor() = default;

    virtual void add_compile_state(std::unique_ptr<CompileState> state) = 0;
    virtual void await_compiles_finished() = 0;

    virtual void for_each_src(std::function<void(CompileState&)> fn) = 0;

    virtual bool is_parallel() = 0;
};

class CodegenBuildPool : public CompileExecutor {
private:
    std::vector<std::unique_ptr<CompileState>> srcs_;
    std::mutex mutex_;
    std::atomic_int outstanding_compiles_ = 0;

    // Thread pool members
    std::vector<std::thread> workers_;
    std::queue<CompileState*> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;

    void worker_loop();

public:
    CodegenBuildPool(int num_threads);
    ~CodegenBuildPool() override;

    void add_compile_state(std::unique_ptr<CompileState> state) override;
    void await_compiles_finished() override;

    void for_each_src(std::function<void(CompileState&)> fn) override;

    bool is_parallel() override { return workers_.size() > 1; }
};

} // namespace docc::compile
