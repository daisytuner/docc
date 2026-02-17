#include "sdfg/transformations/offloading/hip_parallelize_nested_map.h"

#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/hip/hip.h"

namespace sdfg {
namespace transformations {

HIPParallelizeNestedMap::HIPParallelizeNestedMap(structured_control_flow::Map& loop, size_t block_size)
    : loop_(loop), block_size_(block_size) {}

std::string HIPParallelizeNestedMap::name() const { return "HIPParallelizeNestedMap"; }

bool HIPParallelizeNestedMap::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Condition: Check if map is not yet parallelized with HIP
    if (loop_.schedule_type().value() != ScheduleType_Sequential::value()) {
        return false;
    }

    // Condition: Check if parent loop exists
    auto parent = loop_analysis.parent_loop(&loop_);
    if (parent == nullptr) {
        return false;
    }

    // Condition: Check if parent loop is a HIP map, and not Z dimension (final dimension)
    if (auto map = dynamic_cast<structured_control_flow::Map*>(parent)) {
        if (map->schedule_type().value() != hip::ScheduleType_HIP::value()) {
            return false;
        }
        if (hip::ScheduleType_HIP::dimension(map->schedule_type()) == hip::HIPDimension::Z) {
            return false;
        }
        auto parent_indvar = map->indvar();
        auto ancestor = parent;
        while (ancestor) {
            if (auto map_ancestor = dynamic_cast<structured_control_flow::Map*>(ancestor)) {
                parent_indvar = map_ancestor->indvar();
                for (auto& arg : symbolic::atoms(loop_.condition())) {
                    if (symbolic::eq(arg, parent_indvar)) {
                        return false;
                    }
                }
            }
            ancestor = loop_analysis.parent_loop(ancestor);
        }
    } else {
        return false;
    }

    // Condition: Check if current loop starts from 0
    if (!symbolic::eq(loop_.init(), symbolic::zero())) {
        return false;
    }

    // Condition: Loop has a stride of 1
    auto stride = analysis::LoopAnalysis::stride(&loop_);
    if (!symbolic::eq(stride, symbolic::one())) {
        return false;
    }

    return true;
}

void HIPParallelizeNestedMap::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto parent = loop_analysis.parent_loop(&loop_);

    auto parent_dim =
        hip::ScheduleType_HIP::dimension(static_cast<structured_control_flow::Map*>(parent)->schedule_type());

    hip::HIPDimension child_dim;
    if (parent_dim == hip::HIPDimension::X) {
        child_dim = hip::HIPDimension::Y;
    } else if (parent_dim == hip::HIPDimension::Y) {
        child_dim = hip::HIPDimension::Z;
    } else {
        throw InvalidSDFGException("Parent loop is Z dimension, cannot parallelize nested map.");
    }

    auto new_schedule = hip::ScheduleType_HIP::create();
    hip::ScheduleType_HIP::dimension(new_schedule, child_dim);
    hip::ScheduleType_HIP::block_size(new_schedule, symbolic::integer(block_size_));

    builder.update_schedule_type(loop_, new_schedule);
}

void HIPParallelizeNestedMap::to_json(nlohmann::json& j) const {
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        throw std::runtime_error("HIPParallelizeNestedMap transformation does not support for-loops.");
    }
    j["transformation_type"] = this->name();

    // Describe the subgraph in a form compatible with EmbeddingRecorder/EmbeddingReplayer.
    // Keep the existing "loop" and "block_size" fields for backward compatibility.
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", "map"}}}};

    j["parameters"] = {{"block_size", block_size_}};

    j["loop"] = loop_.element_id();
    j["block_size"] = block_size_;
}

HIPParallelizeNestedMap HIPParallelizeNestedMap::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    // Prefer the embedding-compatible representation (subgraph/parameters),
    // but fall back to legacy fields (loop/block_size) if needed.
    size_t loop_id;
    if (j.contains("subgraph")) {
        const auto& subgraph = j.at("subgraph");
        const auto& node_desc = subgraph.at("0");
        loop_id = node_desc.at("element_id").get<size_t>();
    } else {
        loop_id = j.at("loop").get<size_t>();
    }

    size_t block_size;
    if (j.contains("parameters") && j.at("parameters").contains("block_size")) {
        block_size = j.at("parameters").at("block_size").get<size_t>();
    } else {
        block_size = j.at("block_size").get<size_t>();
    }
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::Map*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " is not a loop.");
    }
    return HIPParallelizeNestedMap(*loop, block_size);
}

} // namespace transformations
} // namespace sdfg
