#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {

// Forward declarations for explicit template instantiation
namespace cuda {
class ScheduleType_CUDA;
}
namespace hip {
class ScheduleType_HIP;
}

namespace gpu {

/**
 * @brief GPU map utility functions shared between CUDA and HIP
 *
 * These template functions provide common functionality for GPU map dispatchers.
 * They are parameterized by the ScheduleType class (ScheduleType_CUDA or ScheduleType_HIP).
 *
 * @tparam ScheduleT The schedule type class (cuda::ScheduleType_CUDA or hip::ScheduleType_HIP)
 */

/**
 * @brief Find the block size for nested GPU maps in a given dimension
 * @tparam ScheduleT Schedule type class with value(), dimension(), and block_size() static methods
 * @param node The current map node
 * @param analysis_manager Analysis manager for loop analysis
 * @param dimension GPU dimension (X, Y, or Z)
 * @return Block size expression, or symbolic::one() if not found
 */
template<typename ScheduleT>
symbolic::Expression find_nested_gpu_blocksize(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

/**
 * @brief Find the number of iterations for nested GPU maps in a given dimension
 * @tparam ScheduleT Schedule type class with value() and dimension() static methods
 * @param node The current map node
 * @param analysis_manager Analysis manager for loop and assumptions analysis
 * @param dimension GPU dimension (X, Y, or Z)
 * @return Number of iterations expression, or symbolic::one() if not found
 */
template<typename ScheduleT>
symbolic::Expression find_nested_gpu_iterations(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

/**
 * @brief Check if a map is the outermost GPU map in the loop tree
 * @tparam ScheduleT Schedule type class with value() static method
 * @param node The current map node
 * @param analysis_manager Analysis manager for loop analysis
 * @return true if this is the outermost GPU map, false otherwise
 */
template<typename ScheduleT>
bool is_outermost_gpu_map(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager);

/**
 * @brief Get all induction variables for GPU maps in a given dimension
 * @tparam ScheduleT Schedule type class with value() and dimension() static methods
 * @param node The current map node
 * @param analysis_manager Analysis manager for loop analysis
 * @param dimension GPU dimension (X, Y, or Z)
 * @return Set of induction variable symbols
 */
template<typename ScheduleT>
symbolic::SymbolSet get_gpu_indvars(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

// Extern template declarations to prevent implicit instantiation
extern template symbolic::Expression find_nested_gpu_blocksize<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::Expression find_nested_gpu_blocksize<
    hip::ScheduleType_HIP>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

extern template symbolic::Expression find_nested_gpu_iterations<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::Expression find_nested_gpu_iterations<
    hip::ScheduleType_HIP>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

extern template bool is_outermost_gpu_map<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&);
extern template bool is_outermost_gpu_map<
    hip::ScheduleType_HIP>(structured_control_flow::Map&, analysis::AnalysisManager&);

extern template symbolic::SymbolSet get_gpu_indvars<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::SymbolSet get_gpu_indvars<
    hip::ScheduleType_HIP>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

} // namespace gpu
} // namespace sdfg
