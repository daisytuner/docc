#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace sdfg {
namespace passes {

inline constexpr const char* DOCC_STATISTICS_ENV = "DOCC_STATISTICS";

inline bool statistics_enabled_by_env() {
    const char* val = std::getenv(DOCC_STATISTICS_ENV);
    return val != nullptr && std::string(val) == "1";
}

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

class AnalysisStatistics {
private:
    bool enabled_ = false;
    std::unordered_map<std::string, uint64_t> count_, time_;

public:
    static AnalysisStatistics& instance() {
        static AnalysisStatistics analysis_statistics;
        return analysis_statistics;
    }

    bool enabled() { return enabled_; }
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    void add_analysis(const std::string& name, uint64_t milliseconds);

    std::string summary();
};

class CodegenStatistics {
private:
    bool enabled_ = false;
    std::unordered_map<std::string, uint64_t> count_, time_;

public:
    static CodegenStatistics& instance() {
        static CodegenStatistics codegen_statistics;
        return codegen_statistics;
    }

    bool enabled() { return enabled_; }
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    void add_codegen(const std::string& name, uint64_t milliseconds);

    std::string summary();
};

} // namespace passes
} // namespace sdfg
