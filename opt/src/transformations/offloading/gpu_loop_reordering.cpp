#include "sdfg/transformations/offloading/gpu_loop_reordering.h"

#include <sdfg/transformations/loop_interchange.h>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"

namespace sdfg {
namespace transformations {

GPULoopReordering::GPULoopReordering(structured_control_flow::Map& map_) : map_(map_) {};
bool GPULoopReordering::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_paths = loop_analysis.loop_tree_paths(&this->map_);

    if (loop_paths.size() != 1) {
        return false;
    }

    auto& nested_loops = loop_paths.front();

    if (nested_loops.size() < 2) {
        return false;
    }

    // Criterion: first loop must be a GPU map
    auto first_loop = dynamic_cast<structured_control_flow::Map*>(nested_loops.at(0));
    if (!first_loop) {
        return false;
    }
    if (!gpu::is_gpu_schedule(first_loop->schedule_type())) {
        return false;
    }

    return true;
};

void GPULoopReordering::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_paths = loop_analysis.loop_tree_paths(&this->map_);
    auto& nested_loops = loop_paths.front();

    // Bubble sort permutation indices
    for (size_t i = 0; i < nested_loops.size(); i++) {
        for (size_t j = 0; j < nested_loops.size() - i - 1; j++) {
            auto for_loop = dynamic_cast<structured_control_flow::For*>(nested_loops.at(j));
            auto map = dynamic_cast<structured_control_flow::Map*>(nested_loops.at(j + 1));

            if (!for_loop || !map) {
                continue;
            }
            transformations::LoopInterchange loop_interchange(*for_loop, *map);
            if (loop_interchange.can_be_applied(builder, analysis_manager)) {
                loop_interchange.apply(builder, analysis_manager);
                nested_loops[j] = loop_interchange.new_outer_loop();
                nested_loops[j + 1] = loop_interchange.new_inner_loop();
            }
        }
    }
    analysis_manager.invalidate_all();
};

std::string GPULoopReordering::name() const { return "GPULoopReordering"; };

void GPULoopReordering::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], map_);
}

GPULoopReordering GPULoopReordering::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    size_t first_loop_id;
    const auto& node_desc = j.at("subgraph").at("0");
    first_loop_id = node_desc.at("element_id").get<size_t>();

    auto* first_loop = dynamic_cast<structured_control_flow::Map*>(builder.find_element_by_id(first_loop_id));
    if (!first_loop) {
        throw std::runtime_error("Invalid first_loop_id in GPULoopReordering deserialization");
    }
    return GPULoopReordering(*first_loop);
}

} // namespace transformations
} // namespace sdfg
