#pragma once
#include <condition_variable>
#include <list>
#include <map>
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
    // Tasks whose codegen still needs to run. Codegen is unconstrained, so a plain
    // FIFO queue is enough; running it eagerly keeps gated compiles unblocked.
    std::queue<CompileState*> codegen_queue_;
    // Tasks that finished codegen and are waiting to compile (their compile may still
    // be gated on lower codegen orders).
    std::vector<CompileState*> compile_ready_;
    // Number of states whose codegen has not finished yet, keyed by codegen_order
    // (zero entries erased).
    std::map<int, int> pending_codegen_by_order_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;

    void worker_loop();

    // The following helpers must be called while holding queue_mutex_.
    bool has_pending_codegen_below(int order) const;
    bool has_ready_compile() const;
    CompileState* take_ready_compile();

public:
    CodegenBuildPool(int num_threads);
    ~CodegenBuildPool() override;

    void add_compile_state(std::unique_ptr<CompileState> state) override;
    void await_compiles_finished() override;

    void for_each_src(std::function<void(CompileState&)> fn) override;

    bool is_parallel() override { return workers_.size() > 1; }
};

} // namespace docc::compile
