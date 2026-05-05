#include "docc/compile/codegen_build_pool.h"

namespace docc::compile {

CodegenBuildPool::CodegenBuildPool(int num_threads) {
    if (num_threads > 1) {
        workers_.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&CodegenBuildPool::worker_loop, this);
        }
    }
}

CodegenBuildPool::~CodegenBuildPool() {
    {
        std::lock_guard lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void CodegenBuildPool::worker_loop() {
    while (true) {
        CompileState* task = nullptr;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return stop_ || !work_queue_.empty(); });
            if (stop_ && work_queue_.empty()) {
                return;
            }
            task = work_queue_.front();
            work_queue_.pop();
        }

        task->codegen();
        task->compile();

        if (--outstanding_compiles_ == 0) {
            done_cv_.notify_all();
        }
    }
}

void CodegenBuildPool::add_compile_state(std::unique_ptr<CompileState> state) {
    auto* ptr = state.get();
    {
        std::lock_guard lock(mutex_);
        srcs_.push_back(std::move(state));
        ++outstanding_compiles_;
    }

    if (workers_.empty()) {
        ptr->codegen();
        ptr->compile();
        --outstanding_compiles_;
    } else {
        {
            std::lock_guard lock(queue_mutex_);
            work_queue_.push(ptr);
        }
        queue_cv_.notify_one();
    }
}

void CodegenBuildPool::await_compiles_finished() {
    std::unique_lock lock(queue_mutex_);
    done_cv_.wait(lock, [this] { return outstanding_compiles_.load() == 0; });
}

void CodegenBuildPool::for_each_src(std::function<void(CompileState&)> fn) {
    std::lock_guard lock(mutex_);

    for (auto& src : srcs_) {
        fn(*src);
    }
}

} // namespace docc::compile
