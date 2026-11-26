#include "sdfg/loops/transformations/vectorize_transform.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace loops {
namespace transformations {

VectorizeTransform::VectorizeTransform(structured_control_flow::StructuredLoop& loop) : loop_(loop) {
}

std::string VectorizeTransform::name() const {
    return "VectorizeTransform";
}

bool VectorizeTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (dyn_cast<structured_control_flow::Map*>(&loop_) == nullptr &&
        dyn_cast<structured_control_flow::Reduce*>(&loop_) == nullptr) {
        if (report_) {
            report_->transform_impossible(this, "not a Map or Reduce");
        }
        return false;
    }

    if (loop_.schedule_type().category() != structured_control_flow::ScheduleTypeCategory::None) {
        if (report_) {
            report_->transform_impossible(this, "not sequential");
        }
        return false;
    }

    if (structured_control_flow::ScheduleType_Vectorize::is_set(loop_.schedule_type())) {
        if (report_) {
            report_->transform_impossible(this, "already vectorized");
        }
        return false;
    }

    if (report_) {
        report_->transform_possible(this);
    }
    return true;
}

void VectorizeTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Copy the current schedule (preserving its kind and properties) and annotate it for vectorization.
    auto new_schedule = loop_.schedule_type();
    structured_control_flow::ScheduleType_Vectorize::set(new_schedule);
    builder.update_schedule_type(this->loop_, new_schedule);
    if (report_) {
        report_->transform_applied(this);
    }
}

void VectorizeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

VectorizeTransform VectorizeTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw ::sdfg::transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw ::sdfg::transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return VectorizeTransform(*loop);
}

} // namespace transformations
} // namespace loops
} // namespace sdfg
