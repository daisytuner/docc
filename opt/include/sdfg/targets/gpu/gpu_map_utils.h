#pragma once

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {

// Forward declarations for explicit template instantiation
namespace cuda {
class ScheduleType_CUDA;
}
namespace rocm {
class ScheduleType_ROCM;
}

namespace gpu {

/**
 * @brief GPU map utility functions shared between CUDA and ROCm
 *
 * These template functions provide common functionality for GPU map dispatchers.
 * They are parameterized by the ScheduleType class (ScheduleType_CUDA or ScheduleType_ROCM).
 *
 * @tparam ScheduleT The schedule type class (cuda::ScheduleType_CUDA or rocm::ScheduleType_ROCM)
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

/**
 * @brief Check if a schedule type is a GPU schedule (CUDA or ROCM)
 */
bool is_gpu_schedule(const structured_control_flow::ScheduleType& schedule);

/**
 * @brief Get all GPU Map nodes in a given dimension (in tree traversal order).
 *
 * Unlike get_gpu_indvars, this preserves access to each Map's init / stride
 * so the codegen can emit `indvar = init + thread_flat_id * stride` for
 * arbitrary affine grid loops.
 *
 * @tparam ScheduleT Schedule type class with value() and dimension() static methods
 * @param node The current map node
 * @param analysis_manager Analysis manager for loop analysis
 * @param dimension GPU dimension (X, Y, or Z)
 * @return Vector of Map pointers in the given GPU dimension
 */
template<typename ScheduleT>
std::vector<structured_control_flow::Map*>
get_gpu_maps(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension);

/**
 * @brief Whether parallelizing @p loop to a new GPU grid dimension would race on a
 *        shared container.
 *
 * Adding a grid dimension for @p loop folds the whole enclosing GPU kernel into a
 * single flattened launch: @p loop 's iterations are distributed across the new
 * dimension's threads, while every sibling of @p loop (and of its ancestors up to
 * the outermost GPU map) is re-executed by every thread of that dimension. Since
 * there is no grid-wide barrier inside a kernel, this is unsafe in two ways:
 *
 * 1. Producer/consumer across the fold: @p loop 's writes vary across the new
 *    dimension, so a replicated sibling that reads or writes one of those shared
 *    containers observes another thread's (incomplete) data (RAW/WAR/WAW). This is
 *    the reduction-accumulator -> consumer hazard, e.g. softmax where a reduce writes
 *    `acc[i]` and a following map divides by `acc[i]` in the same kernel.
 * 2. Replicated self-accumulation: a sibling read-modify-write (`acc[i] += x`) is
 *    folded in once per replicated thread and races on the shared buffer.
 *
 * Both hazards only apply to containers that escape the kernel (function
 * arguments/externals or transients living outside, per ArgumentsAnalysis); a
 * container confined to the kernel is privatized per thread and races nothing. For
 * the self-accumulation hazard, a sibling that is itself a GPU map is exempt: it is
 * already parallelized, so codegen maps it onto its own threads instead of
 * replicating it.
 *
 * This helper is schedule-agnostic (a map is "parallelized" iff its schedule is not
 * Sequential), so it serves both the CUDA and ROCm nested-map transformations.
 *
 * @param loop The sequential loop that a nested-parallelization transform wants to promote.
 * @param analysis_manager Analysis manager (LoopAnalysis, Users, ArgumentsAnalysis).
 * @return true if folding @p loop into the kernel would introduce a data race.
 */
bool nested_parallelization_is_unsafe(
    structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager
);

symbolic::Expression get_target_level_dim(TargetLevel target_level, int warp_size);

symbolic::Expression get_target_level_idx(TargetLevel target_level);

// Coarse GPU cooperation level of a target axis.
bool is_grid_level(TargetLevel target_level);
bool is_block_level(TargetLevel target_level);
bool is_warp_level(TargetLevel target_level);

// Induction variables of every offloaded loop (Map or Reduce) at @p target_level
// in @p node's subtree (including @p node itself).
symbolic::SymbolSet target_level_indvars(
    structured_control_flow::StructuredLoop& node, analysis::AnalysisManager& analysis_manager, TargetLevel target_level
);

// Per-level launch schedule for @p node's subtree: for each occupied target level,
// the offloaded schedule with the largest parallel_size, so a single launch
// dimension covers every sibling at that level.
void get_nested_schedule_types(
    structured_control_flow::StructuredLoop& node,
    analysis::AnalysisManager& analysis_manager,
    std::unordered_map<TargetLevel, structured_control_flow::ScheduleType>& output
);

bool nested_warp_dim(structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager);

structured_control_flow::StructuredLoop* find_x_block_owning_warp_level(
    structured_control_flow::StructuredLoop& node, analysis::AnalysisManager& analysis_manager
);

// Extern template declarations to prevent implicit instantiation
extern template symbolic::Expression find_nested_gpu_blocksize<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::Expression find_nested_gpu_blocksize<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

extern template symbolic::Expression find_nested_gpu_iterations<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::Expression find_nested_gpu_iterations<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

extern template bool is_outermost_gpu_map<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&);
extern template bool is_outermost_gpu_map<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map&, analysis::AnalysisManager&);

extern template symbolic::SymbolSet get_gpu_indvars<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template symbolic::SymbolSet get_gpu_indvars<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

extern template std::vector<structured_control_flow::Map*> get_gpu_maps<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);
extern template std::vector<structured_control_flow::Map*> get_gpu_maps<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map&, analysis::AnalysisManager&, GPUDimension);

} // namespace gpu
} // namespace sdfg
