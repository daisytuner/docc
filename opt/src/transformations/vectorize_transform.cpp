#include "sdfg/transformations/vectorize_transform.h"

#include "sdfg/optimization_report/pass_report_consumer.h"

namespace sdfg {
namespace transformations {

VectorizeTransform::VectorizeTransform(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string VectorizeTransform::name() const { return "VectorizeTransform"; }

bool VectorizeTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (dyn_cast<structured_control_flow::Map*>(&loop_) == nullptr &&
        dyn_cast<structured_control_flow::Reduce*>(&loop_) == nullptr) {
        if (report_) report_->transform_impossible(this, "not a Map or Reduce");
        return false;
    }

    auto result = loop_.schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value();

    if (report_) {
        if (result) {
            report_->transform_possible(this);
        } else {
            report_->transform_impossible(this, "not sequential");
        }
    }

    return result;
}

void VectorizeTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    builder.update_schedule_type(this->loop_, vectorize::ScheduleType_Vectorize::create());
    if (report_) report_->transform_applied(this);
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
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return VectorizeTransform(*loop);
}

} // namespace transformations
} // namespace sdfg
