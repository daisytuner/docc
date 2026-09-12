#pragma once

#include <concepts>
#include <string>
#include <utility>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/transformations/transformation_schema.h>

#include <nlohmann/json.hpp>

#include <sdfg/einsum/einsum.h>
#include <sdfg/tiles/transformations/local_storage.h>
#include <sdfg/tiles/transformations/tile_fusion.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_peeling.h>
#include <sdfg/transformations/loop_shift.h>
#include <sdfg/transformations/loop_skewing.h>
#include <sdfg/transformations/loop_split.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/map_collapse.h>
#include <sdfg/transformations/multi_level_tiling.h>
#include <sdfg/transformations/offloading/cuda_offload_transform.h>
#include <sdfg/transformations/offloading/cuda_parallelize_nested_map.h>
#include <sdfg/transformations/offloading/cuda_transform.h>
#include <sdfg/transformations/offloading/gpu_condition_propagation.h>
#include <sdfg/transformations/offloading/gpu_loop_reordering.h>
#include <sdfg/transformations/offloading/gpu_offload_nested_loop.h>
#include <sdfg/transformations/offloading/rocm_offload_transform.h>
#include <sdfg/transformations/offloading/rocm_parallelize_nested_map.h>
#include <sdfg/transformations/offloading/rocm_transform.h>
#include <sdfg/transformations/omp_transform.h>
#include <sdfg/transformations/unroll_transform.h>
#include <sdfg/transformations/vectorize_transform.h>

namespace sdfg {
namespace transformations {

/**
 * @brief Concept for transformations that can be replayed from a JSON description.
 *
 * In addition to being a @ref transformation_concept (i.e. derived from
 * @ref Transformation), a replayable transformation must provide the exact
 * serialization contract used by the replayers:
 *  - a static `from_json(builder, desc)` factory returning the transformation by
 *    value (this also guarantees the type is concrete / non-abstract), and
 *  - a `to_json(json&) const` member overriding the pure-virtual base method.
 *
 * Registering a transformation in @ref dispatch_transformation instantiates this
 * concept, so any type whose `from_json`/`to_json` do not match the required form
 * fails to compile with a clear diagnostic.
 */
template<typename T>
concept replayable_transformation_concept = transformation_concept<T> && requires(
                                                                             builder::StructuredSDFGBuilder& builder,
                                                                             const nlohmann::json& desc,
                                                                             const T& transformation,
                                                                             nlohmann::json& out
                                                                         ) {
    { T::from_json(builder, desc) } -> std::same_as<T>;
    { transformation.to_json(out) } -> std::same_as<void>;
};

namespace detail {

/**
 * @brief Invoke a visitor with a validated transformation type.
 *
 * The @ref replayable_transformation_concept constraint enforces the
 * `from_json`/`to_json` contract at the point of registration.
 */
template<typename T, typename Visitor>
    requires replayable_transformation_concept<T>
decltype(auto) invoke_for(Visitor&& visitor) {
    return std::forward<Visitor>(visitor).template operator()<T>();
}

} // namespace detail

/**
 * @brief Central registry mapping transformation type names to their C++ types.
 *
 * This is the single source of truth that replaces the per-replayer if/else
 * dispatch chains (base @ref Replayer, EmbeddingReplayer, RecordingReplayer).
 * For a given `transformation_name`, the matching transformation type `T` is
 * resolved and the caller-supplied @p visitor is invoked as
 * `visitor.template operator()<T>()`. The visitor decides what to do with the
 * type (e.g. apply and record it), while this function owns the name-to-type
 * mapping and the compile-time validation of each registered type.
 *
 * @tparam Visitor A callable with a templated `operator()` (e.g. a C++20 generic
 *         lambda `[]<typename T>() { ... }`). Its return value is forwarded back
 *         to the caller unchanged.
 * @param transformation_name The registered transformation type name.
 * @param desc The transformation description; used to resolve parameterized
 *        dispatch (e.g. the GPU backend of @c GPUOffloadNestedLoop).
 * @param visitor The visitor invoked with the resolved transformation type.
 * @throws InvalidTransformationDescriptionException if the name is not registered.
 */
template<typename Visitor>
decltype(auto)
dispatch_transformation(const std::string& transformation_name, const nlohmann::json& desc, Visitor&& visitor) {
    if (transformation_name == "LoopTiling") {
        return detail::invoke_for<transformations::LoopTiling>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "MapCollapse") {
        return detail::invoke_for<transformations::MapCollapse>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "MultiLevelTiling") {
        return detail::invoke_for<transformations::MultiLevelTiling>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopDistribute") {
        return detail::invoke_for<transformations::LoopDistribute>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopInterchange") {
        return detail::invoke_for<transformations::LoopInterchange>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LocalStorage") {
        return detail::invoke_for<transformations::LocalStorage>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "TileFusion") {
        return detail::invoke_for<transformations::TileFusion>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopSkewing") {
        return detail::invoke_for<transformations::LoopSkewing>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopShift") {
        return detail::invoke_for<transformations::LoopShift>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopSplit") {
        return detail::invoke_for<transformations::LoopSplit>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "OMPTransform") {
        return detail::invoke_for<transformations::OMPTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "LoopPeeling") {
        return detail::invoke_for<transformations::LoopPeeling>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "VectorizeTransform") {
        return detail::invoke_for<transformations::VectorizeTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "UnrollTransform") {
        return detail::invoke_for<transformations::UnrollTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "CUDATransform") {
        return detail::invoke_for<cuda::CUDATransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "CUDAOffloadTransform") {
        return detail::invoke_for<cuda::CUDAOffloadTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "ROCMOffloadTransform") {
        return detail::invoke_for<rocm::ROCMOffloadTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "CUDAParallelizeNestedMap") {
        return detail::invoke_for<transformations::CUDAParallelizeNestedMap>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "ROCMTransform") {
        return detail::invoke_for<rocm::ROCMTransform>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "ROCMParallelizeNestedMap") {
        return detail::invoke_for<transformations::ROCMParallelizeNestedMap>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "GPUOffloadNestedLoop") {
        const std::string gpu_type = desc.contains("parameters") && desc["parameters"].contains("gpu_type")
                                         ? desc["parameters"]["gpu_type"].get<std::string>()
                                         : cuda::ScheduleType_CUDA_Offload::value();
        if (gpu_type == rocm::ScheduleType_ROCM_Offload::value()) {
            return detail::invoke_for<
                transformations::GPUOffloadNestedLoop<rocm::ScheduleType_ROCM_Offload>>(std::forward<Visitor>(visitor));
        } else {
            return detail::invoke_for<
                transformations::GPUOffloadNestedLoop<cuda::ScheduleType_CUDA_Offload>>(std::forward<Visitor>(visitor));
        }
    } else if (transformation_name == "GPUConditionPropagation") {
        return detail::invoke_for<transformations::GPUConditionPropagation>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "GPUConditionPropagationScope") {
        return detail::invoke_for<transformations::GPUConditionPropagationScope>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "GPULoopReordering") {
        return detail::invoke_for<transformations::GPULoopReordering>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "EinsumLift") {
        return detail::invoke_for<einsum::EinsumLift>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "EinsumExtend") {
        return detail::invoke_for<einsum::EinsumExtend>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "EinsumExpand") {
        return detail::invoke_for<einsum::EinsumPromotion>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "Einsum2Dot") {
        return detail::invoke_for<einsum::Einsum2Dot>(std::forward<Visitor>(visitor));
    } else if (transformation_name == "Einsum2Gemm") {
        return detail::invoke_for<einsum::Einsum2Gemm>(std::forward<Visitor>(visitor));
    } else {
        throw transformations::InvalidTransformationDescriptionException("Unknown transformation: " + transformation_name);
    }
}

/**
 * @brief Verify that a transformation's `from_json` and `to_json` are consistent inverses.
 *
 * This is the semantic counterpart to @ref validate_transformation_schema: while
 * the schema validates the *shape* of a single description, this check validates
 * that the registry's `from_json` factory and the transformation's `to_json`
 * serialization agree with each other. Concretely it:
 *   1. reconstructs the transformation from @p desc via `T::from_json`,
 *   2. re-serializes it via `to_json` (the canonical description),
 *   3. asserts the canonical description satisfies the shared schema,
 *   4. asserts the reconstructed `transformation_type` matches @p desc, and
 *   5. asserts the round-trip is a stable fixed point
 *      (`to_json(from_json(x)) == to_json(from_json(to_json(from_json(x))))`),
 *      which fails if `from_json` drops or misreads any field that `to_json` writes.
 *
 * @tparam T The transformation type (must satisfy @ref replayable_transformation_concept).
 * @param builder The SDFG builder whose elements the description refers to.
 * @param desc The transformation description to round-trip.
 * @param error_out Populated with a human-readable reason when verification fails.
 * @return true if `from_json` and `to_json` are consistent for @p desc.
 */
template<typename T>
    requires replayable_transformation_concept<T>
bool verify_round_trip(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc, std::string& error_out) {
    // 1) Reconstruct from the (possibly enriched) description and re-serialize.
    T first(T::from_json(builder, desc));
    nlohmann::json canonical;
    first.to_json(canonical);

    // 2) The output of to_json must itself satisfy the shared schema.
    if (!validate_transformation_schema(canonical, error_out)) {
        error_out = "to_json() produced a schema-invalid description: " + error_out;
        return false;
    }

    // 3) The reconstructed type must match the description that selected it.
    if (desc.contains("transformation_type") && desc.at("transformation_type") != canonical.at("transformation_type")) {
        error_out = "transformation_type changed across round-trip: description '" +
                    desc.at("transformation_type").get<std::string>() + "' vs to_json '" +
                    canonical.at("transformation_type").get<std::string>() + "'";
        return false;
    }

    // 4) from_json/to_json must be a stable fixed point: any field written by
    //    to_json but not read back by from_json would perturb this comparison.
    T second(T::from_json(builder, canonical));
    nlohmann::json canonical_again;
    second.to_json(canonical_again);
    if (canonical != canonical_again) {
        error_out = "from_json/to_json is not a stable round-trip: " + canonical.dump() +
                    " != " + canonical_again.dump();
        return false;
    }

    return true;
}

/**
 * @brief Verify `from_json`/`to_json` consistency for the transformation named in @p desc.
 *
 * Resolves the transformation type through the registry and delegates to
 * @ref verify_round_trip. This ties every registered `from_json` factory to its
 * `to_json` serialization through a single, name-driven entry point.
 *
 * @param builder The SDFG builder whose elements the description refers to.
 * @param desc The transformation description (must contain a string `transformation_type`).
 * @param error_out Populated with a human-readable reason when verification fails.
 * @return true if `from_json` and `to_json` are consistent for @p desc.
 */
inline bool verify_transformation_round_trip(
    builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc, std::string& error_out
) {
    if (!desc.contains("transformation_type") || !desc.at("transformation_type").is_string()) {
        error_out = "description is missing a string 'transformation_type'";
        return false;
    }
    const auto transformation_name = desc.at("transformation_type").get<std::string>();
    std::string local_error;
    const bool ok = dispatch_transformation(transformation_name, desc, [&]<typename T>() -> bool {
        return verify_round_trip<T>(builder, desc, local_error);
    });
    if (!ok) {
        error_out = std::move(local_error);
    }
    return ok;
}

} // namespace transformations
} // namespace sdfg
