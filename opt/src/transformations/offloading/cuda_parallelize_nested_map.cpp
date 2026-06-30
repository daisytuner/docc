#include "sdfg/transformations/offloading/cuda_parallelize_nested_map.h"

#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace transformations {

CUDAParallelizeNestedMap::CUDAParallelizeNestedMap(structured_control_flow::Map& loop, size_t block_size)
    : loop_(loop), block_size_(block_size) {}

std::string CUDAParallelizeNestedMap::name() const { return "CUDAParallelizeNestedMap"; }

bool CUDAParallelizeNestedMap::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Condition: Check if map is not yet parallelized with CUDA
    if (loop_.schedule_type().value() != ScheduleType_Sequential::value()) {
        return false;
    }

    // Condition: Check if parent loop exists
    auto parent = loop_analysis.parent_loop(&loop_);
    if (parent == nullptr) {
        return false;
    }

    // Condition: Check if parent loop is a CUDA map, and not Z dimension (final dimension)
    if (auto map = dynamic_cast<structured_control_flow::Map*>(parent)) {
        if (map->schedule_type().value() != cuda::ScheduleType_CUDA::value()) {
            return false;
        }
        if (cuda::ScheduleType_CUDA::dimension(map->schedule_type()) == cuda::CUDADimension::Z) {
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

    // Note: arbitrary `init` and `stride` are permitted. The CUDA dispatcher
    // emits `<map.indvar> = init + thread_flat_id * stride`, so the body sees
    // the natural strided value; `num_iterations()` accounts for both when
    // computing the grid geometry.

    // Condition: Resulting CUDA grid dimension must not exceed hardware limits.
    // Y and Z grid dimensions are limited to 65535.
    auto num_iters = loop_.num_iterations();
    if (!num_iters.is_null() && SymEngine::is_a<SymEngine::Integer>(*num_iters)) {
        int64_t iters = SymEngine::down_cast<const SymEngine::Integer&>(*num_iters).as_int();
        int64_t block = static_cast<int64_t>(block_size_);
        int64_t grid_size = (iters + block - 1) / block;

        constexpr int64_t max_grid_dim_yz = 65535;
        if (grid_size > max_grid_dim_yz) {
            return false;
        }
    }

    return true;
}

void CUDAParallelizeNestedMap::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto parent = loop_analysis.parent_loop(&loop_);

    auto parent_dim =
        cuda::ScheduleType_CUDA::dimension(static_cast<structured_control_flow::Map*>(parent)->schedule_type());

    cuda::CUDADimension child_dim;
    if (parent_dim == cuda::CUDADimension::X) {
        child_dim = cuda::CUDADimension::Y;
    } else if (parent_dim == cuda::CUDADimension::Y) {
        child_dim = cuda::CUDADimension::Z;
    } else {
        throw InvalidSDFGException("Parent loop is Z dimension, cannot parallelize nested map.");
    }

    auto new_schedule = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(new_schedule, child_dim);
    cuda::ScheduleType_CUDA::block_size(new_schedule, symbolic::integer(block_size_));

    builder.update_schedule_type(loop_, new_schedule);
}

void CUDAParallelizeNestedMap::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["block_size"] = block_size_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

CUDAParallelizeNestedMap CUDAParallelizeNestedMap::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    // Prefer the embedding-compatible representation (subgraph/parameters),
    // but fall back to legacy fields (loop/block_size) if needed.
    const auto& subgraph = j.at("subgraph");
    const auto& node_desc = subgraph.at("0");
    size_t loop_id = node_desc.at("element_id").get<size_t>();

    size_t block_size = j.at("parameters").at("block_size").get<size_t>();
    auto loop = dynamic_cast<structured_control_flow::Map*>(builder.find_element_by_id(loop_id));
    if (!loop) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " is not a loop.");
    }
    return CUDAParallelizeNestedMap(*loop, block_size);
}

} // namespace transformations
} // namespace sdfg
