#pragma once

#include <cstddef>
#include <optional>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg {
namespace tiles {

/**
 * @file copy_widen.h
 * @brief Widen a cooperative shared-staging copy block into a single vector /
 *        async transfer.
 *
 * A cooperative copy emits, per lane, a scalar `src[..] --assign--> buf[..]`
 * inside a thread-scheduled coverage map. When the copied run is contiguous and
 * aligned, several consecutive elements per lane can move in one wide transfer:
 * the coverage map is strided by the coalescing `factor` and the scalar block is
 * rewritten into an address-taking reference-memlet pair feeding a copy node.
 * Shared between SoftwarePipelining (async cp.async) and LocalStorage (plain
 * synchronous vectorization).
 */

/// Which transfer primitive the rewrite lowers the copy to.
enum class CopyTransfer {
    CpAsync, ///< CpAsyncCopyNode (+ pipeline commit/wait, owned by the caller)
    VectorSync, ///< VectorCopyNode: a single synchronous 4/8/16-byte vector move
};

/**
 * @brief Rewrite one cooperative copy @p block into a widened transfer.
 *
 * @p allow_vectorize opts fp32 into its 16-byte (float4) coalescing; narrow
 * elements (<4B) always coalesce to reach a legal >=4-byte width. For
 * @ref CopyTransfer::CpAsync the semantics are unchanged from the original
 * in-pipeline lowering (a legal 4/8/16-byte cp.async is always emitted, even at
 * factor 1). For @ref CopyTransfer::VectorSync the block is only rewritten when
 * coalescing actually widens the transfer (factor > 1); an un-widenable copy is
 * left as the original scalar tasklet.
 *
 * @p implementation_type is stamped onto the emitted copy node so the owning
 * target's dispatcher (CUDA/ROCm) is selected at codegen; the caller resolves it
 * from the enclosing tile target.
 *
 * @return the emitted transfer width in bytes, or std::nullopt when @p block was
 *         left unchanged.
 */
std::optional<size_t> rewrite_cooperative_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    bool allow_vectorize,
    CopyTransfer transfer,
    const data_flow::ImplementationType& implementation_type
);

/**
 * @brief Widen an already-lowered cooperative copy node (VectorCopy or CpAsync)
 *        further, re-striding its (possibly already-strided) coverage map.
 *
 * Unlike @ref rewrite_cooperative_copy (which lowers a scalar tasklet), this
 * operates on a copy that a prior pass already turned into a node — e.g. a
 * minimal-width cp.async emitted by SoftwarePipelining — and coalesces it to the
 * widest legal vector width. @p node lives in @p node_block; @p current_bytes is
 * its present transfer width. Returns the new width, or std::nullopt when it
 * could not widen (left unchanged).
 */
std::optional<size_t> widen_existing_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& node_block,
    data_flow::LibraryNode& node,
    size_t current_bytes
);

} // namespace tiles
} // namespace sdfg
