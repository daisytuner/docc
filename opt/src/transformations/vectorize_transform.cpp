#include "sdfg/transformations/vectorize_transform.h"

#include "sdfg/optimization_report/pass_report_consumer.h"

namespace sdfg {
namespace transformations {

VectorizeTransform::VectorizeTransform(structured_control_flow::Map& map) : map_(map) {}

std::string VectorizeTransform::name() const { return "VectorizeTransform"; }

bool VectorizeTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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

void VectorizeTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    builder.update_schedule_type(this->map_, vectorize::ScheduleType_Vectorize::create());
    if (report_) report_->transform_applied(this);
}

void VectorizeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};
    j["transformation_type"] = this->name();
}

VectorizeTransform VectorizeTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(map_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::Map*>(element);

    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " is not a Map.");
    }

    return VectorizeTransform(*loop);
}

} // namespace transformations
} // namespace sdfg
