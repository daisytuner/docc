#include "sdfg/parallelization/transformations/loop_parallelization.h"

#include "sdfg/parallelization/passes/auto_parallelization.h"

namespace sdfg {
namespace transformations {

LoopParallelization::LoopParallelization(structured_control_flow::For& loop) : loop_(loop) {
}

std::string LoopParallelization::name() const {
    return "LoopParallelization";
}

bool LoopParallelization::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (dynamic_cast<structured_control_flow::Sequence*>(loop_.get_parent()) == nullptr) {
        return false;
    }
    std::vector<structured_control_flow::ReductionInfo> reductions;
    parallelization::AutoParallelization classification;
    return classification.classify(builder, analysis_manager, loop_, reductions) ==
           parallelization::AutoParallelization::Classification::Map;
}

void LoopParallelization::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    map_ = &builder.convert_for(parent, loop_);
    analysis_manager.invalidate_all();
}

void LoopParallelization::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", "for"}}}};
}

LoopParallelization LoopParallelization::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::For*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " is not a For.");
    }
    return LoopParallelization(*loop);
}

structured_control_flow::Map* LoopParallelization::map() const {
    return map_;
}

} // namespace transformations
} // namespace sdfg
