#include "sdfg/transformations/omp_transform.h"

#include <stdexcept>

#include "sdfg/optimization_report/pass_report_consumer.h"

namespace sdfg {
namespace transformations {

OMPTransform::OMPTransform(structured_control_flow::Map& map) : map_(map) {}

std::string OMPTransform::name() const { return "OMPTransform"; }

bool OMPTransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto result = map_.schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value();

    if (report_) {
        if (result) {
            report_->transform_possible(this);
        } else {
            report_->transform_impossible(this, "not sequential");
        }
    }
    return result;
}

void OMPTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    builder.update_schedule_type(this->map_, omp::ScheduleType_OMP::create());
    if (report_) report_->transform_applied(this);
}

void OMPTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};
    j["transformation_type"] = this->name();
}

OMPTransform OMPTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(map_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(map_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::Map*>(element);

    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(map_id) + " is not a Map.");
    }

    return OMPTransform(*loop);
}

} // namespace transformations
} // namespace sdfg
