#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"


namespace sdfg {
namespace analysis {

struct DataRwFlags {
    bool is_explicit_input;
    bool is_input;
    bool is_output;

    DataRwFlags(bool is_explicit_input, bool is_input, bool is_output)
        : is_explicit_input(is_explicit_input), is_input(is_input), is_output(is_output) {}
    DataRwFlags() : is_explicit_input(false), is_input(false), is_output(false) {}
    DataRwFlags(const DataRwFlags& copy) = default;
    ~DataRwFlags() = default;
    DataRwFlags& operator=(const DataRwFlags& other) = default;

    void found_explicit_read() {
        is_explicit_input = true;
        is_input = true;
    }

    void found_explicit_write() {
        is_input = true;
        is_output = true;
    }

    void found_analysis_escape() {
        is_input = true;
        is_output = true;
        is_explicit_input = true;
    }

    void merge(DataRwFlags& other) {
        is_explicit_input |= other.is_explicit_input;
        is_input |= other.is_input;
        is_output |= other.is_output;
    }
};

struct RegionArgument : public DataRwFlags {
    bool is_scalar;
    bool is_ptr;

    RegionArgument(bool is_explicit_input, bool is_input, bool is_output, bool scalar, bool ptr)
        : DataRwFlags(is_explicit_input, is_input, is_output), is_scalar(scalar), is_ptr(ptr) {}
    RegionArgument(DataRwFlags rwFlags, bool scalar, bool ptr) : DataRwFlags(rwFlags), is_scalar(scalar), is_ptr(ptr) {}

    RegionArgument() : is_scalar(false), is_ptr(false) {}

    RegionArgument(const RegionArgument& copy) = default;

    ~RegionArgument() = default;

    RegionArgument& operator=(const RegionArgument& other) = default;
};

class ArgumentsAnalysis : public Analysis {
private:
    std::unordered_map<structured_control_flow::ControlFlowNode*, std::map<std::string, RegionArgument>> node_arguments_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, std::unordered_set<std::string>> node_locals_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, bool> node_inferred_types_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, symbolic::Expression>>
        argument_sizes_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, symbolic::Expression>>
        argument_element_sizes_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, bool> known_sizes_;

    void find_arguments_and_locals(
        analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
    );

    void collect_arg_sizes(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        bool allow_dynamic_sizes_,
        bool do_not_throw
    );

public:
    ArgumentsAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "ArgumentsAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    const std::map<std::string, RegionArgument>&
    arguments(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node);

    const std::unordered_set<std::string>&
    locals(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node);

    bool inferred_types(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node);

    const std::unordered_map<std::string, symbolic::Expression>& argument_sizes(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        bool allow_dynamic_sizes_
    );

    const std::unordered_map<std::string, symbolic::Expression>& argument_element_sizes(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        bool allow_dynamic_sizes_
    );

    bool argument_size_known(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        bool allow_dynamic_sizes_
    );
};

} // namespace analysis
} // namespace sdfg
