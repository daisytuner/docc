#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sdfg {
namespace codegen {

class ArgumentRegistry {
private:
    mutable std::mutex compile_mutex_;
    mutable std::mutex linker_mutex_;
    std::unordered_map<std::string, std::vector<std::string>> compile_args_map_;
    std::unordered_map<std::string, std::vector<std::string>> linker_args_map_;

public:
    static ArgumentRegistry& instance() {
        static ArgumentRegistry registry;
        return registry;
    }

    void register_compile_args(std::string target, std::vector<std::string> args) {
        std::lock_guard<std::mutex> lock(compile_mutex_);
        if (compile_args_map_.find(target) != compile_args_map_.end()) {
            throw std::runtime_error(
                "Library node dispatcher already registered for library node code: " + std::string(target)
            );
        }
        compile_args_map_[target] = std::move(args);
    }

    std::vector<std::string> get_compile_args(std::string target) const {
        auto it = compile_args_map_.find(target);
        if (it != compile_args_map_.end()) {
            return it->second;
        }
        return {};
    }

    size_t size_compile_args() const {
        return compile_args_map_.size();
    }

    void register_linker_args(std::string target, std::vector<std::string> args) {
        std::lock_guard<std::mutex> lock(linker_mutex_);
        if (linker_args_map_.find(target) != linker_args_map_.end()) {
            throw std::runtime_error(
                "Library node dispatcher already registered for library node code: " + std::string(target)
            );
        }
        linker_args_map_[target] = std::move(args);
    }

    std::vector<std::string> get_linker_args(std::string target) const {
        auto it = linker_args_map_.find(target);
        if (it != linker_args_map_.end()) {
            return it->second;
        }
        return {};
    }

    size_t size_linker_args() const {
        return linker_args_map_.size();
    }
};

} // namespace codegen
} // namespace sdfg
