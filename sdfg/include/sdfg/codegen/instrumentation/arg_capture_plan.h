#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

class CaptureVarPlan {
public:
    const bool capture_input;
    const bool capture_output;
    const int arg_idx;
    const bool is_external;
    const symbolic::Expression size;
    const bool is_scalar;

    const sdfg::types::PrimitiveType inner_type;

    CaptureVarPlan(
        bool capture_input,
        bool capture_output,
        int arg_idx,
        bool is_external,
        sdfg::types::PrimitiveType inner_type,
        const symbolic::Expression size,
        bool is_scalar
    );
};

class ArgCapturePlan {
private:
    StructuredSDFG& sdfg_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes_;

    static bool add_capture_plan(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        const std::string& var_name,
        analysis::RegionArgument region_arg,
        int arg_idx,
        bool is_external,
        std::unordered_map<std::string, CaptureVarPlan>& plan
    );

public:
    ArgCapturePlan(
        StructuredSDFG& sdfg,
        const std::unordered_map<
            const structured_control_flow::ControlFlowNode*,
            std::unordered_map<std::string, CaptureVarPlan>>& nodes
    )
        : sdfg_(sdfg), nodes_(nodes) {
    }

    bool is_empty() const {
        return nodes_.empty();
    }

    bool should_instrument(const structured_control_flow::ControlFlowNode& node) const;

    void begin_instrumentation(
        const structured_control_flow::ControlFlowNode& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension
    ) const;

    void end_instrumentation(
        const structured_control_flow::ControlFlowNode& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension
    ) const;

    void emit_arg_captures(
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const std::unordered_map<std::string, CaptureVarPlan>& plan,
        bool after,
        std::string element_id
    ) const;

    static std::unique_ptr<ArgCapturePlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<ArgCapturePlan> root(StructuredSDFG& sdfg);

    static std::unique_ptr<ArgCapturePlan> outermost_loops_plan(StructuredSDFG& sdfg);

    static std::unordered_map<std::string, CaptureVarPlan> create_capture_plan(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node
    );
};

} // namespace codegen
} // namespace sdfg
