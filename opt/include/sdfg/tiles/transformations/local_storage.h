#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"
#include "sdfg/tiles/analysis/reduction_analysis.h"
#include "sdfg/tiles/analysis/tile_analysis.h"
#include "sdfg/tiles/buffer_layout.h"
#include "sdfg/tiles/locality.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Local-storage transformation.
 *
 * Relocates a compile-time-bounded region (a *tile*) of a large pointer container
 * into a small, densely packed, contiguous local buffer whose storage space is
 * derived from the schedule (see @ref tiles::LocalityPlan::required_space).
 *
 */
class LocalStorage : public Transformation {
public:
    /// Normalized tile description handed from can_be_applied() to apply().
    struct TileInfo {
        /// Overapproximated integer extents per delinearized dimension.
        std::vector<symbolic::Expression> dimensions;
        /// Tile min indices per dimension (bases for index subtraction).
        std::vector<symbolic::Expression> bases;
        /// Tile max valid indices per dimension (from MemoryTile::max_subset), e.g.
        /// min(_s0-1, base+extent-1, ...). Used to element-predicate global copies
        /// so the over-approximated tile never touches out-of-bounds memory.
        std::vector<symbolic::Expression> maxes;
        /// Layout strides from MemoryLayoutAnalysis (original re-linearization).
        std::vector<symbolic::Expression> strides;
        /// Layout offset from MemoryLayoutAnalysis.
        symbolic::Expression offset = symbolic::integer(0);

        /// Indices of the varying (extent > 1) dims, in dimension order. The
        /// degenerate extent-1 dims collapse to their base and carry no buffer axis.
        std::vector<size_t> varying_dims() const;
        /// Extents of the varying dims, in dimension order.
        std::vector<symbolic::Expression> varying_sizes() const;
        /// The source geometry as a tiles @ref tiles::Layout over the varying tile
        /// coordinate (shape = varying extents, strides = varying dims' strides,
        /// offset = layout offset with each dim's `base*stride` folded in). So
        /// `source_layout().apply_coords(tile_indices)` == @ref original_subset.
        tiles::Layout source_layout() const;
        /// Container linear address for per-varying-dim @p tile_indices (extent-1
        /// dims use their base). Pure; unit-testable.
        std::vector<symbolic::Expression> original_subset(const std::vector<symbolic::Expression>& tile_indices) const;
        /// Per-varying-dim local tile index (@p access_subset[d] - base[d]) for a
        /// body access. Pure; unit-testable.
        std::vector<symbolic::Expression> local_index(const std::vector<symbolic::Expression>& access_subset) const;

        /// Container linear address for the *full flat* tile (slots folded in):
        /// substitutes each @p slot_indvars[s] with @p slot_values[s] (the indvar's
        /// value for the delinearized slot index), yielding the gather address for a
        /// lane-contiguous cooperative copy. Pure; unit-testable.
        std::vector<symbolic::Expression> flat_original_subset(
            const std::vector<symbolic::Expression>& slot_indvars,
            const std::vector<symbolic::Expression>& slot_values,
            const std::vector<symbolic::Expression>& tile_indices
        ) const;
    };

    /// True if any library node in @p loop's body has side effects (may touch
    /// memory outside the tracked memlets, so localization must bail).
    static bool has_side_effect(structured_control_flow::StructuredLoop& loop);

private:
    structured_control_flow::StructuredLoop& loop_;
    const data_flow::AccessNode& access_node_;
    std::string container_;
    std::string local_name_; ///< Name of the created local buffer (valid after apply())
    types::StorageType storage_type_; ///< Storage type for the local buffer (derived by can_be_applied)
    TileInfo tile_info_; ///< Populated by can_be_applied()
    tiles::LocalityPlan plan_; ///< Schedule classification (populated by can_be_applied)
    bool swizzle_layout_ = false; ///< XOR-swizzle the NV_Shared inner index instead of padding it
    bool lane_contiguous_ = false; ///< Lay the NV_Shared tile thread-linearly (flat, no slots) for the CDNA async
                                   ///< global->LDS DMA, which writes lane-contiguous from a wave-uniform base.
    std::unordered_set<const data_flow::Memlet*> group_memlets_; ///< Memlets in the selected tile group
    std::vector<structured_control_flow::Reduce*> reduce_retargets_; ///< non-cooperative Reduce nodes to retarget in
                                                                     ///< apply()
    std::vector<structured_control_flow::Reduce*> grid_reduce_owners_; ///< grid-parallel ancestor Reduce nodes to
                                                                       ///< demote to Map (atomic merge)
    bool atomic_merge_ = false; ///< copy-out atomically merges the per-block partial into the global accumulator
    bool container_read_ = false; ///< Container is read in the loop (set by can_be_applied)
    bool container_written_ = false; ///< Container is written in the loop (set by can_be_applied)

    /// Copy the tile in before the loop iff the container is read there.
    bool needs_copy_in() const { return container_read_; }

    /// Copy the tile back after the loop iff the container is written there.
    bool needs_copy_out() const { return container_written_; }

    /// Hard capacity guard: max scalar slots the local buffer may occupy.
    size_t max_tile_elements() const { return 1u << 16; }

    /// Element-predicate a global copy: AND over varying dims of `base[d] +
    /// tile_index <= maxes[d]`, so the over-approximated tile skips out-of-bounds
    /// global elements. @p tile_indices are per varying dim. boolTrue if unbounded.
    /// Conjuncts provably always-true under @p assums (given @p params) are
    /// dropped, so a fully-covering tile emits no guard at all.
    symbolic::Condition boundary_guard(
        const data_flow::Subset& tile_indices,
        const symbolic::SymbolSet& params = {},
        const symbolic::Assumptions& assums = {}
    ) const;

    /// Private/CPU path: a nested sequential copy nest around the loop (copy-in
    /// when @p writeback is false, copy-out when true).
    void emit_private_copy(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& parent,
        const tiles::PackedBuffer& buffer,
        const types::IType& buffer_type,
        const types::IType& pointer_type,
        bool writeback
    );

    /// Cooperative GPU path: a flattened copy-in Map carrying the cooperative
    /// dim's offload schedule, followed by a barrier (read-only, no writeback).
    /// @p slot_indices are the per-thread buffer-slot indices (threadIdx.<axis>);
    /// @p leading_barrier adds a pre-copy barrier for re-staged (per-thread) tiles.
    void emit_cooperative_copy_in(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& parent,
        const tiles::PackedBuffer& buffer,
        const types::IType& buffer_type,
        const types::IType& pointer_type,
        const std::vector<symbolic::Expression>& slot_indices,
        bool leading_barrier
    );

    /// Stage the tile once at the top of the localized GPU map's body (a
    /// block-scheduled copy map + trailing barrier), so the block consumers below
    /// read the shared buffer. Used for the enclosing-cooperative case.
    void emit_enclosing_cooperative_copy_in(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const tiles::PackedBuffer& buffer,
        const types::IType& buffer_type,
        const types::IType& pointer_type
    );

    /// Lane-contiguous cooperative path (CDNA async global->LDS): a full-block
    /// thread-linear copy into a flat buffer, so `dst = buf[flat_tid + BLK*iter]`
    /// is lane-contiguous (required by global_load_lds, which ignores per-lane
    /// destinations). Gathers via TileInfo::flat_original_subset from the
    /// delinearized flat index. @p slot_* describe the folded per-thread dims.
    void emit_lane_contiguous_copy_in(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& parent,
        const tiles::PackedBuffer& buffer,
        const types::IType& buffer_type,
        const types::IType& pointer_type,
        const std::vector<symbolic::Expression>& slot_sizes,
        const std::vector<symbolic::Expression>& slot_indvars,
        const std::vector<symbolic::Expression>& slot_inits,
        const std::vector<symbolic::Expression>& slot_strides
    );

    /// Redirect every container access in the loop body to the local buffer,
    /// prefixed by the per-thread @p slot_indices.
    void rewrite_body(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const tiles::PackedBuffer& buffer,
        const types::IType& buffer_type,
        const std::vector<symbolic::Expression>& slot_indices
    );

public:
    /**
     * @brief Construct a local-storage transformation for @p access_node's
     *        container within @p loop. Direction and storage space are derived by
     *        can_be_applied().
     *
     * @param swizzle_layout A bank-conflict-free NV_Shared tile uses an XOR
     *        *swizzle* of the inner index instead of *padding* — no wasted columns
     *        and the layout `ldmatrix` needs. Requires a power-of-two inner block;
     *        falls back to padding otherwise.
     * @param lane_contiguous The NV_Shared tile is laid out flat and thread-linear
     *        (slots folded, no padding), staged by a full-block cooperative copy.
     *        Required by the CDNA async global->LDS DMA (`global_load_lds`).
     */
    LocalStorage(
        structured_control_flow::StructuredLoop& loop,
        const data_flow::AccessNode& access_node,
        bool swizzle_layout = false,
        bool lane_contiguous = false
    )
        : loop_(loop), access_node_(access_node), container_(access_node.data()),
          storage_type_(types::StorageType::CPU_Stack()), swizzle_layout_(swizzle_layout),
          lane_contiguous_(lane_contiguous) {}

    std::string name() const override { return "LocalStorage"; }

    /**
     * @brief Precondition check: verifies the container is a pointer used in the
     *        loop, resolves a single tile group with integer extents, derives the
     *        storage space, and populates tile_info_ — rejecting any schedule the
     *        apply path cannot yet handle.
     */
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /// Allocate the buffer, emit the derived copies, and redirect loop-body
    /// accesses to the buffer.
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /// JSON serialization.
    void to_json(nlohmann::json& j) const override;

    /// JSON deserialization.
    static LocalStorage from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// Name of the created local buffer (valid after apply()).
    const std::string& local_container() const { return local_name_; }

    /// Tile info (valid after can_be_applied() returns true).
    const TileInfo& tile_info() const { return tile_info_; }

    /// Storage type of the local buffer.
    const types::StorageType& storage_type() const { return storage_type_; }
};

} // namespace transformations
} // namespace sdfg
