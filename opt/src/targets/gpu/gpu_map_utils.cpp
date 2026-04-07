#include "sdfg/targets/gpu/gpu_map_utils.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace gpu {

template<typename ScheduleT>
symbolic::Expression find_nested_gpu_blocksize(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);

    // Check for repeated dimensions in loop tree paths
    auto loop_tree_paths = loop_analysis.loop_tree_paths(&node);
    for (auto& path : loop_tree_paths) {
        bool foundX = false;
        bool foundY = false;
        bool foundZ = false;
        for (auto& loop : path) {
            if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
                if (map->schedule_type().value() == ScheduleT::value()) {
                    auto dim = ScheduleT::dimension(map->schedule_type());
                    if (dim == GPUDimension::X) {
                        if (foundX) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated X dimension");
                        }
                        foundX = true;
                    } else if (dim == GPUDimension::Y) {
                        if (foundY) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated Y dimension");
                        }
                        foundY = true;
                    } else if (dim == GPUDimension::Z) {
                        if (foundZ) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated Z dimension");
                        }
                        foundZ = true;
                    }
                }
            }
        }
    }

    // Find block size for the requested dimension
    for (auto loop : loops) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() != ScheduleT::value() &&
                map->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
                throw InvalidSDFGException("Nested map in GPU kernel not GPU or Sequential");
            }

            if (map->schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value()) {
                continue;
            }

            if (ScheduleT::dimension(map->schedule_type()) == dimension) {
                return ScheduleT::block_size(map->schedule_type());
            }
        }
    }
    return symbolic::one();
}

template<typename ScheduleT>
symbolic::Expression find_nested_gpu_iterations(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);

    for (auto loop : loops) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() != ScheduleT::value() &&
                map->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
                throw InvalidSDFGException("Nested map in GPU kernel not GPU or Sequential");
            }
            if (map->schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value()) {
                continue;
            }
            if (ScheduleT::dimension(map->schedule_type()) != dimension) {
                continue;
            }

            auto init = map->init();
            if (!symbolic::eq(init, symbolic::zero())) {
                throw InvalidSDFGException("Init is not zero");
            }

            auto stride = map->stride();
            if (!symbolic::eq(stride, symbolic::one())) {
                throw InvalidSDFGException("Stride is not one");
            }

            auto num_iterations = map->num_iterations();
            if (num_iterations.is_null()) {
                throw InvalidSDFGException("Cannot determine number of iterations for nested map in GPU kernel");
            }
            return num_iterations;
        }
    }
    return symbolic::one();
}

template<typename ScheduleT>
bool is_outermost_gpu_map(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& loop_tree = loop_analysis.loop_tree();
    structured_control_flow::ControlFlowNode* ancestor = loop_tree.at(&node);
    while (ancestor != nullptr) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(ancestor)) {
            if (map->schedule_type().value() == ScheduleT::value()) {
                return false;
            }
        }
        ancestor = loop_tree.at(ancestor);
    }
    return true;
}

template<typename ScheduleT>
symbolic::SymbolSet get_gpu_indvars(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    symbolic::SymbolSet indvars;
    for (const auto& loop : loops) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() == ScheduleT::value()) {
                if (ScheduleT::dimension(map->schedule_type()) == dimension) {
                    indvars.insert(map->indvar());
                }
            }
        }
    }
    return indvars;
}

// Explicit template instantiations for CUDA
template symbolic::Expression find_nested_gpu_blocksize<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template symbolic::Expression find_nested_gpu_iterations<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template bool is_outermost_gpu_map<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager);

template symbolic::SymbolSet get_gpu_indvars<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

// Explicit template instantiations for ROCM
template symbolic::Expression find_nested_gpu_blocksize<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template symbolic::Expression find_nested_gpu_iterations<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template bool is_outermost_gpu_map<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager);

template symbolic::SymbolSet get_gpu_indvars<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

} // namespace gpu
} // namespace sdfg
