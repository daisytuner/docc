#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/flop_analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace codegen {

class InstrumentationPlan {
protected:
    StructuredSDFG& sdfg_;
    std::unordered_set<const Element*> nodes_;

public:
    InstrumentationPlan(StructuredSDFG& sdfg, const std::unordered_set<const Element*>& nodes)
        : sdfg_(sdfg), nodes_(nodes) {}

    InstrumentationPlan(const InstrumentationPlan& other) = delete;
    InstrumentationPlan(InstrumentationPlan&& other) = delete;

    InstrumentationPlan& operator=(const InstrumentationPlan& other) = delete;
    InstrumentationPlan& operator=(InstrumentationPlan&& other) = delete;

    bool is_empty() const { return nodes_.empty(); }

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

    void insert(const Element* node) { nodes_.insert(node); }

    static std::unique_ptr<InstrumentationPlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<InstrumentationPlan> outermost_loops_plan(StructuredSDFG& sdfg);
};

} // namespace codegen
} // namespace sdfg
