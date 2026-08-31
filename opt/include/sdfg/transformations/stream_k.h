#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Stream-K work decomposition: replace a static output-tile grid whose
 *        blocks each own one tile with a fixed persistent grid whose blocks each
 *        walk an equal contiguous slice of the flattened (tile x k-panel)
 *        iteration space, merging partially-owned tiles via the reduction.
 *
 * ## Shape it targets (post-offload, post-LocalStorage)
 * The canonical GEMM nest after grid offload + LocalStorage:
 *   grid[ iTile, jTile ] {                 // parallel output-tile band
 *     C_reg = 0                            // LocalStorage register accumulator
 *     Reduce{Add, C}( for kPanel {         // associative reduction over K panels
 *        coop-load A,B -> shared; barrier; microkernel(C_reg += ...); barrier
 *     } )
 *     writeback C_reg -> C                 // sole-owner store today
 *   }
 *
 * ## Rewrite it produces
 *   grid1d[ NBLOCKS ]  (NBLOCKS = blocks_per_cu * device_multiprocessor_count) {
 *     iter     = blockIdx * TOTAL / NBLOCKS         // TOTAL = num_tiles * panels
 *     iter_end = (blockIdx+1) * TOTAL / NBLOCKS
 *     while iter < iter_end {
 *        tile      = iter / panels
 *        k_start   = iter % panels
 *        seg_end   = min(iter_end, (tile+1)*panels) - tile*panels
 *        (row0,col0) = decode(tile)
 *        C_reg = 0
 *        for kPanel in [k_start, seg_end): <existing coop-load + microkernel>
 *        owns_full = (k_start == 0 && seg_end == panels)
 *        owns_full ? store(C_reg -> C) : Reduce{Add}(C_reg -> C)   // conditional merge
 *        iter = tile*panels + seg_end
 *     }
 *   }
 * Requires C pre-zeroed for the atomic partials (host memset or a prologue).
 *
 * ## can_be_applied criteria
 *  1. @p grid_loop_ is (or roots) a GPU-offloaded parallel output-tile band.
 *  2. Its body contains a Reduce node whose operator is associative with a
 *     hardware atomic (Add). This is the fold axis; detected via the Reduce node
 *     type, not body pattern-matching.
 *  3. The reduction accumulator element type has a device atomic-add (fp32/fp64/
 *     int; fp16/bf16 only where supported) -- else reject (or workspace+fixup v2).
 *  4. Uniform, compile-time-constant panel count (K/BK) equal across tiles, and
 *     constant tile counts (M/BM, N/BN): needed to form TOTAL and the decode.
 *  5. The accumulator writeback targets global C (no fused downstream consumer on
 *     the same launch that would race the partial tiles).
 * Any failure -> can_be_applied returns false (conservative, like the others).
 *
 * ## Generality
 *  Applies to any associative-reduction-over-tiles: GEMM, GEMV, batched/grouped
 *  GEMM, conv-as-GEMM (implicit GEMM), tensor contractions with a reduction axis.
 *  Excludes: non-associative "reductions" (scan/argmax w/o atomic), dtypes without
 *  atomics, RAGGED reduction extent across tiles (needs a per-tile offset table,
 *  v2), and strict bit-reproducibility (atomic float ordering; use workspace+fixup).
 */
class StreamK : public Transformation {
    // Outermost GPU-offloaded parallel output-tile loop (the band converted to a
    // 1-D persistent worker grid).
    structured_control_flow::StructuredLoop& grid_loop_;

    // Fixed persistent grid size (absolute block count).
    size_t num_blocks_;

public:
    explicit StreamK(structured_control_flow::StructuredLoop& grid_loop, size_t num_blocks = 336);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static StreamK from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
