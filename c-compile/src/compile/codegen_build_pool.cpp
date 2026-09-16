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
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void CodegenBuildPool::worker_loop() {
    while (true) {
        CompileState* codegen_task = nullptr;
        CompileState* compile_task = nullptr;
        {
            std::unique_lock lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_ || !codegen_queue_.empty() || has_ready_compile(); });
            if (stop_ && codegen_queue_.empty() && compile_ready_.empty()) {
                return;
            }

            // Always prefer codegen work: codegen is unconstrained, and finishing it
            // is what unblocks the order-gated compiles. This also guarantees a worker
            // never blocks while codegen it may be waiting for is still queued.
            if (!codegen_queue_.empty()) {
                codegen_task = codegen_queue_.front();
                codegen_queue_.pop();
            } else {
                compile_task = take_ready_compile();
                if (compile_task == nullptr) {
                    // Only gated compiles remain; their codegen dependencies are being
                    // produced by other workers, so wait for progress and retry.
                    continue;
                }
            }
        }

        if (codegen_task != nullptr) {
            codegen_task->codegen();
            {
                std::lock_guard lock(queue_mutex_);
                auto it = pending_codegen_by_order_.find(codegen_task->codegen_order());
                if (it != pending_codegen_by_order_.end() && --(it->second) == 0) {
                    pending_codegen_by_order_.erase(it);
                }
                compile_ready_.push_back(codegen_task);
            }
            // Finishing a codegen may open gates and adds a compilable task.
            cv_.notify_all();
            continue;
        }

        compile_task->compile();

        if (--outstanding_compiles_ == 0) {
            std::lock_guard lock(queue_mutex_);
            done_cv_.notify_all();
        }
    }
}

bool CodegenBuildPool::has_pending_codegen_below(int order) const {
    return !pending_codegen_by_order_.empty() && pending_codegen_by_order_.begin()->first < order;
}

bool CodegenBuildPool::has_ready_compile() const {
    for (auto* task : compile_ready_) {
        if (!has_pending_codegen_below(task->compile_min_order())) {
            return true;
        }
    }
    return false;
}

CompileState* CodegenBuildPool::take_ready_compile() {
    for (auto it = compile_ready_.begin(); it != compile_ready_.end(); ++it) {
        if (!has_pending_codegen_below((*it)->compile_min_order())) {
            CompileState* task = *it;
            compile_ready_.erase(it);
            return task;
        }
    }
    return nullptr;
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
            ++pending_codegen_by_order_[ptr->codegen_order()];
            codegen_queue_.push(ptr);
        }
        cv_.notify_one();
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
