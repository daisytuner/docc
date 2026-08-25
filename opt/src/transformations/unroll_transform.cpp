#include "sdfg/transformations/unroll_transform.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

UnrollTransform::UnrollTransform(structured_control_flow::StructuredLoop& loop) : loop_(loop) {};

std::string UnrollTransform::name() const { return "UnrollTransform"; };

/// True if `expr` is a strictly positive integer constant.
static bool is_positive_int(const symbolic::Expression& expr) {
    return expr != SymEngine::null && SymEngine::is_a<SymEngine::Integer>(*expr) &&
           SymEngine::rcp_static_cast<const SymEngine::Integer>(expr)->as_int() > 0;
}

bool UnrollTransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Only a provably constant trip count can be fully unrolled.
    return is_positive_int(loop_.num_iterations());
};

void UnrollTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Copy the current schedule (preserving its kind) and annotate it for unrolling.
    auto new_schedule = loop_.schedule_type();
    structured_control_flow::ScheduleType_Unroll::set(new_schedule);
    builder.update_schedule_type(loop_, new_schedule);

    analysis_manager.invalidate_all();
};

void UnrollTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
};

UnrollTransform UnrollTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    return UnrollTransform(*loop);
};

} // namespace transformations
} // namespace sdfg
