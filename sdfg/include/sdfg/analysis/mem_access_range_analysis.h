#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class MemAccessRange {
    friend class MemAccessRangesBuilder;

private:
    const std::string name_;
    bool saw_read_;
    bool saw_write_;
    bool undefined_;
    std::vector<std::pair<symbolic::Expression, symbolic::Expression>> dims_;

public:
    MemAccessRange(
        const std::string& name,
        bool saw_read,
        bool saw_write,
        bool undefined,
        const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>&& dims
    );

    MemAccessRange(const MemAccessRange& other)
        : name_(other.name_), saw_read_(other.saw_read_), saw_write_(other.saw_write_), undefined_(other.undefined_),
          dims_(other.dims_) {}

    MemAccessRange(MemAccessRange&& other) noexcept
        : name_(std::move(other.name_)), saw_read_(other.saw_read_), saw_write_(other.saw_write_),
          undefined_(other.undefined_), dims_(std::move(other.dims_)) {}

    const std::string& get_name() const;

    bool saw_read() const;
    bool saw_write() const;
    bool is_undefined() const;

    const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>& dims() const;
};

class MemAccessRanges : public Analysis {
    friend class AnalysisManager;

private:
    // Graph representation
    graph::Graph graph_;

    std::unordered_map<structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, MemAccessRange>>
        ranges_;

    analysis::AnalysisManager* analysis_manager_;

    void run(structured_control_flow::ControlFlowNode& node, std::unordered_set<std::string> target_container);

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    MemAccessRanges(StructuredSDFG& sdfg);

    std::string name() const override { return "MemAccessRanges"; }

    const MemAccessRange* get(const std::string& varName) const;

    const MemAccessRange*
    get(const std::string& varName,
        structured_control_flow::ControlFlowNode& node,
        std::unordered_set<std::string> target_container);
};

} // namespace analysis
} // namespace sdfg
