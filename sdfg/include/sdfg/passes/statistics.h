#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sdfg {
namespace passes {

inline constexpr const char* DOCC_STATISTICS_ENV = "DOCC_STATISTICS";

inline bool statistics_enabled_by_env() {
    const char* val = std::getenv(DOCC_STATISTICS_ENV);
    return val != nullptr && std::string(val) == "1";
}

int statistics_mode_env();

// Generic base class for hierarchical statistics collection.
// Captures a tree of named scopes; each scope stores an enter/exit timestamp
// and a map of uint64 metrics (e.g. "loopCount"). The class takes over the
// timing: enter() and exit() record the timestamps automatically.
// Not thread-safe by design: one instance is expected to be used per thread.
class HierarchicalStatistics {
public:
    struct Node {
        std::string scope_type;
        std::string name;
        std::chrono::high_resolution_clock::time_point enter_time;
        std::chrono::high_resolution_clock::time_point exit_time;
        std::unordered_map<std::string, uint64_t> metrics;
        Node* parent = nullptr;
        std::vector<std::unique_ptr<Node>> children;

        uint64_t duration_ms() const;
    };

protected:
    std::vector<std::unique_ptr<Node>> roots_;
    Node* current_ = nullptr;

    enum class RemapAction { Proceed, NoDescend, Skip };

    static void print_node_self(std::ostream& stream, const Node& node, int depth);

    void print_node(std::ostream& stream, const HierarchicalStatistics::Node& node, int depth);

    virtual RemapAction custom_print_node(std::ostream& out, const Node&, int depth);

    std::string report(const std::string& title);

public:
    virtual ~HierarchicalStatistics() = default;

    // Open a new scope as a child of the currently active scope.
    Node* enter_scope(const std::string& name, const std::string& scope_type);

    // Close the currently active scope.
    void exit_scope();

    // Set/accumulate a metric on the currently active scope.
    void set_metric(const std::string& key, uint64_t value);
    void add_metric(const std::string& key, uint64_t value);

    void clear();
};

class CompileStatistics : public HierarchicalStatistics {
public:
    enum class ReportLevel {
        PipelineTotalsOnly = 0,
        SummarizePipelineContentsAndAnalysis = 1,
        All = 2,
    };

protected:
    static bool enabled_;

    static constexpr const char* ANALYSIS_SCOPE = "Analysis";
    static constexpr const char* ANALYSIS_MGR_SCOPE = "AnalysisMgr";
    static constexpr const char* PASS_SCOPE = "Pass";
    static constexpr const char* PIPELINE_SCOPE = "Pipeline";
    static constexpr const char* STAGE_SCOPE = "Stage";
    ReportLevel report_detail_ = ReportLevel::SummarizePipelineContentsAndAnalysis;

public:
    static CompileStatistics& instance() {
        static CompileStatistics analysis_statistics;
        return analysis_statistics;
    }

    // Static wrappers that no-op when statistics are disabled.
    static void enter_analysis_if_enabled(const std::string& name) {
        if (CompileStatistics::enabled_) {
            auto& inst = instance();
            inst.enter_analysis(name);
        }
    }

    void enter_analysis(const std::string& name) { enter_scope(name, ANALYSIS_SCOPE); }

    void exit_analysis() { exit_scope(); }

    static void exit_analysis_if_enabled() {
        if (CompileStatistics::enabled_) {
            auto& inst = instance();
            inst.exit_analysis();
        }
    }

    static void enter_pass_if_enabled(const std::string& name) {
        if (enabled_) {
            auto& inst = instance();
            inst.enter_pass(name);
        }
    }

    void enter_pass(const std::string& name) { enter_scope(name, PASS_SCOPE); }

    static void exit_pass_if_enabled() {
        if (enabled_) {
            auto& inst = instance();
            inst.exit_pass();
        }
    }

    void exit_pass() { exit_scope(); }

    static void enter_pipeline_if_enabled(const std::string& name) {
        if (enabled_) {
            auto& inst = instance();
            inst.enter_pipeline(name);
        }
    }

    void enter_pipeline(const std::string& name) { enter_scope(name, PIPELINE_SCOPE); }

    static void exit_pipeline_if_enabled() {
        if (enabled_) {
            auto& inst = instance();
            inst.exit_pipeline();
        }
    }

    void exit_pipeline() { exit_scope(); }

    static void enter_stage_if_enabled(const std::string& name) {
        if (enabled_) {
            auto& inst = instance();
            inst.enter_stage(name);
        }
    }

    void enter_stage(const std::string& name) { enter_scope(name, STAGE_SCOPE); }

    static void exit_stage_if_enabled() {
        if (enabled_) {
            auto& inst = instance();
            inst.exit_stage();
        }
    }

    void exit_stage() { exit_scope(); }

    static void set_metric_if_enabled(const std::string& key, uint64_t value) {
        if (enabled_) {
            auto& inst = instance();
            inst.set_metric(key, value);
        }
    }

    static void add_metric_if_enabled(const std::string& key, uint64_t value) {
        if (enabled_) {
            auto& inst = instance();
            inst.add_metric(key, value);
        }
    }

    static bool enabled() { return enabled_; }
    static void enable() { enabled_ = true; }
    static void disable() { enabled_ = false; }

    std::string summary();

    std::string report(ReportLevel detail);

protected:
    RemapAction custom_print_node(std::ostream& out, const Node& node, int depth) override;

    // Collapse a pipeline scope into a single level: passes are summed per name,
    // any other scope type is aggregated into an "other" bucket per scope type.
    static void summarize_pipeline_passes(std::ostream& out, const Node& node, int depth);
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
