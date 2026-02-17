#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace sdfg {
namespace passes {

class PassStatistics {
private:
    bool enabled_ = false;
    std::unordered_map<std::string, uint64_t> sdfg_count_, sdfg_time_, structured_sdfg_count_, structured_sdfg_time_;

public:
    static PassStatistics& instance() {
        static PassStatistics pass_statistics;
        return pass_statistics;
    }

    bool enabled() { return enabled_; }
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    void add_sdfg_pass(const std::string& name, uint64_t milliseconds);
    void add_structured_sdfg_pass(const std::string& name, uint64_t milliseconds);

    std::string summary();
};

class PipelineStatistics {
private:
    bool enabled_ = false;
    std::unordered_map<std::string, uint64_t> sdfg_count_, sdfg_time_, structured_sdfg_count_, structured_sdfg_time_;

public:
    static PipelineStatistics& instance() {
        static PipelineStatistics pipeline_statistics;
        return pipeline_statistics;
    }

    bool enabled() { return enabled_; }
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    void add_sdfg_pipeline(const std::string& name, uint64_t milliseconds);
    void add_structured_sdfg_pipeline(const std::string& name, uint64_t milliseconds);

    std::string summary();
};

} // namespace passes
} // namespace sdfg
