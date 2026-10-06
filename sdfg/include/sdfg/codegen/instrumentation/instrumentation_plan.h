#pragma once

#include <optional>
#include <unordered_map>
#include <vector>
#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/metadata/loop_provenance.h"
#include "sdfg/metadata/rpc_optimization.h"
#include "sdfg/options.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace codegen {

// Global opt-in for wrapping instrumented regions in an adaptive sampling loop.
inline constexpr OptionKey<bool> INSTRUMENTATION_ADAPTIVE_SAMPLING{"instrumentation.adaptive_sampling"};

class InstrumentationPlan {
protected:
    StructuredSDFG& sdfg_;
    std::unordered_set<const Element*> nodes_;
    std::unordered_map<const Element*, ElementId> logical_region_ids_;
    struct GroupSpan {
        ElementId source_loop_id;
        const structured_control_flow::Sequence* sequence;
        std::vector<const structured_control_flow::ControlFlowNode*> members;
    };
    std::vector<GroupSpan> group_spans_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, size_t> group_start_by_node_;
    std::unordered_set<const structured_control_flow::ControlFlowNode*> grouped_members_;
    // When false, leaving_instrumentation_function does not emit finalize_all so a
    // harness can resolve pending events once (e.g. after a warm sampling batch)
    // instead of paying a host sync on every invocation of an SDFG.
    bool emit_finalize_all_;
    // When true, each instrumented region is wrapped in an adaptive sampling loop.
    bool sampling_;

public:
    InstrumentationPlan(
        StructuredSDFG& sdfg,
        const std::unordered_set<const Element*>& nodes,
        bool emit_finalize_all = true,
        bool sampling = false
    )
        : sdfg_(sdfg), nodes_(nodes), emit_finalize_all_(emit_finalize_all), sampling_(sampling) {
    }

    InstrumentationPlan(const InstrumentationPlan& other) = delete;
    InstrumentationPlan(InstrumentationPlan&& other) = delete;

    InstrumentationPlan& operator=(const InstrumentationPlan& other) = delete;
    InstrumentationPlan& operator=(InstrumentationPlan&& other) = delete;

    bool is_empty() const {
        return nodes_.empty();
    }

    bool sampling() const {
        return sampling_;
    }

    bool should_instrument(const Element& node) const;

    void begin_instrumentation(
        const Element& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const InstrumentationInfo& info
    ) const;

    void end_instrumentation(
        const Element& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const InstrumentationInfo& info
    ) const;

    void leaving_instrumentation_function(PrettyPrinter& stream, LanguageExtension& language_extension) const;

    void insert(const Element* node) {
        nodes_.insert(node);
    }

    std::optional<ElementId> logical_region_id(const Element& node) const {
        auto it = logical_region_ids_.find(&node);
        return it == logical_region_ids_.end() ? std::nullopt : std::optional<ElementId>(it->second);
    }

    std::optional<ElementId> source_loop_id(const Element& node) const {
        return metadata::source_loop_id(node);
    }

    std::optional<nlohmann::json> expected_performance(const Element& node) const {
        const auto result = metadata::rpc_optimization(node);
        return result.has_value() ? result->expected_performance : std::nullopt;
    }

    std::optional<double> vector_distance(const Element& node) const {
        const auto result = metadata::rpc_optimization(node);
        return result.has_value() ? result->vector_distance : std::nullopt;
    }

    const GroupSpan* group_span_starting_at(const structured_control_flow::ControlFlowNode& node) const {
        auto it = group_start_by_node_.find(&node);
        return it == group_start_by_node_.end() ? nullptr : &group_spans_[it->second];
    }

    bool is_group_member(const structured_control_flow::ControlFlowNode& node) const {
        return grouped_members_.find(&node) != grouped_members_.end();
    }

    static std::unique_ptr<InstrumentationPlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<InstrumentationPlan>
    outermost_loops_plan(StructuredSDFG& sdfg, bool emit_finalize_all = true, bool sampling = false);

    static std::unique_ptr<InstrumentationPlan>
    provenance_grouped_outermost_loops_plan(StructuredSDFG& sdfg, bool emit_finalize_all = true, bool sampling = false);
};

} // namespace codegen
} // namespace sdfg
