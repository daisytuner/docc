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
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Standalone local-storage transformation.
 *
 * Relocates a compile-time-bounded region (a *tile*) of a large pointer
 * container into a small, densely packed, contiguous local buffer whose storage
 * space is *derived from the schedule* (see LocalityPlan / derive_storage). The
 * loop-body accesses are redirected to that buffer, and memory consistency is
 * preserved by copying the tile in before the loop and/or out after it.
 *
 * The three properties that make this a *performance* transformation:
 *   - **bounded**   — every tile extent resolves to an integer constant.
 *   - **dense**     — the buffer is row-major `product(extents)` even when the
 *                     original access is strided; the repacking is the win.
 *   - **consistent**— copy-in and/or copy-out chosen by read/write presence.
 *
 * Direction is *derived*, not a policy: a read of the container implies copy-in,
 * a write implies copy-out, and both together give a read-modify-write buffer.
 * The storage space is likewise derived from the enclosing parallel schedule.
 *
 * @note v1 (CPU) intentionally does NOT support mixed group / split-node
 *       access nodes: if the selected access node carries memlets from more
 *       than one tile group, the transformation cleanly rejects.
 * @note A schedule that would require block-shared memory (a cooperative GPU
 *       read tile) is recognised by derive_storage() but its apply path is not
 *       yet implemented, so such cases currently reject.
 * @note The tile is an over-approximated bounding box, so halo/stencil
 *       accesses over-copy but remain correct.
 */
class LocalStorage : public Transformation {
public:
    /// Normalized tile description handed from can_be_applied() to apply().
    struct TileInfo {
        /// Overapproximated integer extents per delinearized dimension.
        std::vector<symbolic::Expression> dimensions;
        /// Tile min indices per dimension (bases for index subtraction).
        std::vector<symbolic::Expression> bases;
        /// Layout strides from MemoryLayoutAnalysis (original re-linearization).
        std::vector<symbolic::Expression> strides;
        /// Layout offset from MemoryLayoutAnalysis.
        symbolic::Expression offset = symbolic::integer(0);

        /// Indices of the varying (extent > 1) dims, in dimension order. The
        /// degenerate extent-1 dims collapse to their base and carry no buffer axis.
        std::vector<size_t> varying_dims() const;
        /// Extents of the varying dims, in dimension order.
        std::vector<symbolic::Expression> varying_sizes() const;
        /// Container linear address for per-varying-dim @p tile_indices (extent-1
        /// dims use their base). Pure; unit-testable.
        std::vector<symbolic::Expression> original_subset(const std::vector<symbolic::Expression>& tile_indices) const;
        /// Per-varying-dim local tile index (@p access_subset[d] - base[d]) for a
        /// body access. Pure; unit-testable.
        std::vector<symbolic::Expression> local_index(const std::vector<symbolic::Expression>& access_subset) const;
    };

    /// Dense row-major local buffer laid out as [per-thread slot dims] ++ [tile
    /// dims]. Pure index arithmetic, no SDFG state — unit-testable in isolation.
    struct TileBuffer {
        /// Per-thread buffer-prefix dims (one slot per GPU thread along a
        /// per-thread parallel dim); empty for the non-mixed cooperative/private case.
        std::vector<symbolic::Expression> slot_sizes;
        /// Cooperative / varying tile dims (the actual staged region).
        std::vector<symbolic::Expression> tile_sizes;

        /// Total scalar slots = product(slot_sizes) * product(tile_sizes).
        symbolic::Expression total_size() const;
        /// Product of the tile dims only (one per-thread slot's worth).
        symbolic::Expression tile_total_size() const;
        /// Buffer offset of a per-thread slot: row-major over slot_sizes, scaled
        /// by tile_total_size so each slot owns a contiguous tile block.
        symbolic::Expression slot_offset(const std::vector<symbolic::Expression>& slot_indices) const;
        /// Row-major linear index over [slot_indices ++ tile_indices].
        symbolic::Expression linearize(
            const std::vector<symbolic::Expression>& slot_indices, const std::vector<symbolic::Expression>& tile_indices
        ) const;
        /// Decompose a flat tile index (0..product(tile_sizes)) into per-tile-dim
        /// indices (row-major).
        std::vector<symbolic::Expression> delinearize_tile(const symbolic::Expression& flat) const;
    };

    /// How a container is accessed within a loop, read straight off the
    /// dataflow (memlet types), independent of any pointer-weak Users analysis.
    struct AccessSummary {
        bool reads = false; ///< a computational read of the container
        bool writes = false; ///< a computational write of the container
        bool aliased = false; ///< the pointer escapes, is overwritten, or is captured by a library node
    };

    /**
     * @brief Schedule-derived classification of how the tile relates to the
     *        enclosing parallel loop nest — the basis for deriving the storage
     *        space, copy pattern, and synchronization.
     *
     * For every enclosing parallel loop we record whether the localized tile is
     *   - **per-thread** — the loop's indvar appears in a tile base, so each
     *     iteration addresses a distinct slice and owns a private copy; or
     *   - **cooperative** — the indvar is absent from every base, so all
     *     iterations along that dim share one tile and must stage it together.
     *
     * Each GPU dim additionally carries its parallelism *level* (grid block,
     * thread block, or warp). The coarsest cooperative level determines the
     * required memory space: cooperation across blocks (grid) needs global
     * memory, cooperation within a block (block) needs shared memory, and warp
     * cooperation is served by shuffles (no buffer). A cooperative CPU dim
     * cannot be backed by a private stack at all; the purely per-thread /
     * sequential case maps to a thread-private stack buffer.
     */
    struct LocalityPlan {
        /// GPU parallelism level of a dim, coarsest (Grid) to finest (Warp).
        enum class Level { Grid, Block, Warp };

        struct Dim {
            symbolic::Symbol indvar; ///< the parallel loop's induction variable
            bool cooperative = false; ///< indvar absent from every tile base
            bool is_gpu = false; ///< GPU (CUDA/ROCM) offloader schedule
            Level level = Level::Block; ///< GPU parallelism level (meaningful iff is_gpu)
            gpu::TargetLevel target_level = gpu::TargetLevel::X_BLOCK; ///< GPU axis (for threadIdx slotting)
            symbolic::Integer parallel_size = symbolic::integer(0); ///< parallel width (0 on CPU)
            bool needs_sync = false; ///< schedule requires nested synchronization
        };

        std::vector<Dim> dims; ///< enclosing parallel loops, innermost-first
        bool loop_is_outermost = false; ///< the localized loop is the outermost loop
        bool loop_is_gpu = false; ///< the localized loop itself is GPU-scheduled
        bool has_gpu_descendant = false; ///< a GPU map lives inside the loop body
        /// The localized loop is itself a GPU map and a block-scheduled loop in its
        /// body consumes the tile: stage once per block into shared, reused by the
        /// (sibling) consumers below (e.g. fused softmax: cache the row, then
        /// max-reduce / sum-reduce / normalize all read from shared).
        bool enclosing_cooperative = false;

        /// True when a GPU-scheduled loop encloses us (we are inside a device kernel).
        bool inside_gpu_kernel() const {
            for (const auto& d : dims)
                if (d.is_gpu) return true;
            return false;
        }
        /// True when any cooperative GPU dim exists (threads share the tile).
        bool has_gpu_cooperative() const {
            for (const auto& d : dims)
                if (d.is_gpu && d.cooperative) return true;
            return false;
        }
        /// True when a cooperative GPU dim at @p level exists.
        bool has_cooperative_at(Level level) const {
            for (const auto& d : dims)
                if (d.is_gpu && d.cooperative && d.level == level) return true;
            return false;
        }
        /// True when a cooperative non-GPU parallel dim exists (no private stack fits).
        bool has_cpu_cooperative() const {
            for (const auto& d : dims)
                if (!d.is_gpu && d.cooperative) return true;
            return false;
        }
        /// The GPU per-thread dims (indvar in a tile base) — each owns a buffer slot.
        std::vector<Dim> gpu_per_thread_dims() const {
            std::vector<Dim> out;
            for (const auto& d : dims)
                if (d.is_gpu && !d.cooperative) out.push_back(d);
            return out;
        }
        /// The GPU cooperative dims (the shared/copy parallelism axes).
        std::vector<Dim> gpu_cooperative_dims() const {
            std::vector<Dim> out;
            for (const auto& d : dims)
                if (d.is_gpu && d.cooperative) out.push_back(d);
            return out;
        }
    };

    /**
     * @brief Classify @p container's accesses within @p loop using the shared
     *        pointer-analysis infrastructure (PointerEscape/Overwrite/Used
     *        analyzers over the loop body).
     *
     * `aliased` is set when the pointer escapes (address leak / return), is
     * overwritten/swapped, or is passed to a library node that may capture it
     * (`pointer_access_type` missing or not `no_capture`) — any of which lets
     * the container's memory be reached outside the memlets we rewrite.
     */
    static AccessSummary
    summarize(const StructuredSDFG& sdfg, structured_control_flow::StructuredLoop& loop, const std::string& container);

    /**
     * @brief True if any library node in @p loop's body has side effects.
     *
     * Such a node may read or write memory (including the container) outside the
     * memlets we track, so localization must bail. Mirrors the SideEffectFinder
     * used by the offloading/fusion passes.
     */
    static bool has_side_effect(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief The CPU precondition: @p group is a real tile whose extents are all
     *        compile-time integer constants.
     *
     * A null group, an empty tile, or any symbolic/unbounded (null) extent means
     * no constantly bounded tile exists and the transformation must reject.
     */
    static bool is_constant_bounded(const analysis::MemoryTileGroup* group) {
        if (!group) {
            return false;
        }
        auto extents = group->tile.extents_approx();
        if (extents.empty()) {
            return false;
        }
        for (auto& extent : extents) {
            if (extent.is_null() || !SymEngine::is_a<SymEngine::Integer>(*extent)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief All tile groups MemoryLayoutAnalysis formed for @p container at the
     *        @p loop scope (nullptr if none).
     */
    static const std::vector<analysis::MemoryTileGroup>* tile_groups(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        analysis::AnalysisManager& analysis_manager
    ) {
        return analysis_manager.get<analysis::MemoryLayoutAnalysis>().tile_groups(loop, container);
    }

    /**
     * @brief Number of scalar slots a packed buffer for @p group would occupy
     *        (product of extents) — the allocation and copy-in/out volume.
     *
     * @return The product; a compile-time integer iff is_constant_bounded(group),
     *         SymEngine::null if any extent is unbounded, 0 for a null group.
     */
    static symbolic::Expression tile_element_count(const analysis::MemoryTileGroup* group) {
        if (!group) {
            return symbolic::integer(0);
        }
        symbolic::Expression count = symbolic::integer(1);
        for (auto& extent : group->tile.extents_approx()) {
            if (extent.is_null()) {
                return SymEngine::null;
            }
            count = symbolic::mul(count, extent);
        }
        return symbolic::simplify(count);
    }

    /**
     * @brief The localizable tile of @p container at @p loop (container-anchored).
     *
     * Returns the sole tile group for @p container at @p loop iff the container
     * forms EXACTLY ONE group there AND every one of its memlets in the loop body
     * belongs to that group (no unanalyzable or split-across-groups accesses).
     * Otherwise nullptr.
     *
     * Anchoring on the container rather than a single access node means all of
     * its access nodes are localized together as one coherent tile, which is what
     * makes wholesale rewriting safe.
     */
    static const analysis::MemoryTileGroup* tile(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        analysis::AnalysisManager& analysis_manager
    );

    /**
     * @brief Classify the tile against the enclosing parallel loop nest.
     *
     * Walks @p loop's ancestor chain, and for each genuinely parallel loop
     * records whether the tile is per-thread or cooperative (see LocalityPlan),
     * tagging GPU dims with their parallelism level, width, and sync requirement.
     */
    static LocalityPlan build_locality_plan(
        structured_control_flow::StructuredLoop& loop,
        const TileInfo& tile_info,
        analysis::AnalysisManager& analysis_manager
    );

    /// Storage space derived from a LocalityPlan.
    enum class Locality {
        Reject, ///< the schedule cannot be safely localized
        Private, ///< thread-private / sequential buffer (CPU_Stack; registers when tiny)
        Shared, ///< block-shared GPU buffer (NV_Shared)
        Global, ///< grid-wide GPU buffer (NV_Global)
    };

    /**
     * @brief Derive the required storage space from the schedule.
     *
     * - A cooperative non-GPU parallel dim cannot be served by a private stack
     *   → Reject.
     * - A cooperative *write* is a reduction owned by the Reduce node + reduce
     *   dispatcher → Reject (only read-only cooperative tiles localize here).
     * - A cooperative GPU read tile inside a kernel (and not the outermost loop)
     *   maps to the coarsest cooperative level: grid → Global, block → Shared;
     *   a warp-only cooperative tile is served by shuffles (no buffer) → Reject.
     * - A host-level loop that itself is GPU-scheduled or wraps a GPU kernel is
     *   not a localization site → Reject.
     * - Otherwise the tile is per-thread / sequential → Private.
     */
    static Locality derive_storage(const LocalityPlan& plan, bool container_written);

    /**
     * @brief Whether @p container is the accumulator of a Reduce enclosing,
     *        nested within, or equal to @p loop.
     *
     * Detects the reduction-accumulator relationship. Localizing such a
     * container is permitted only for a *non-cooperative* (sequential /
     * per-thread) Reduce at or below @p loop — see @ref collect_reduction_owners
     * — where LocalStorage privatizes the accumulator and apply() retargets the
     * Reduce's descriptor to the local buffer. A cooperatively-combined
     * (GPU-offloaded) Reduce is owned by the reduce dispatcher and must not be
     * localized here.
     */
    static bool is_reduction_accumulator(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        analysis::AnalysisManager& analysis_manager
    );

    /**
     * @brief Collect the non-cooperative Reduce nodes (@p loop itself or a
     *        descendant) that reduce into @p container, for accumulator
     *        privatization.
     *
     * @return false if a *cooperative* (GPU-combined) Reduce or an *ancestor*
     *         Reduce owns @p container — those cannot be safely localized at
     *         @p loop and must reject. Otherwise fills @p out with the owning
     *         non-cooperative Reduce nodes (possibly empty) and returns true.
     */
    static bool collect_reduction_owners(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::Reduce*>& out
    );

private:
    structured_control_flow::StructuredLoop& loop_;
    const data_flow::AccessNode& access_node_;
    std::string container_;
    std::string local_name_; ///< Name of the created local buffer (valid after apply())
    types::StorageType storage_type_; ///< Storage type for the local buffer (derived by can_be_applied)
    TileInfo tile_info_; ///< Populated by can_be_applied()
    LocalityPlan plan_; ///< Schedule classification (populated by can_be_applied)
    std::unordered_set<const data_flow::Memlet*> group_memlets_; ///< Memlets in the selected tile group
    std::vector<structured_control_flow::Reduce*> reduce_retargets_; ///< non-cooperative Reduce nodes to retarget in
                                                                     ///< apply()
    bool container_read_ = false; ///< Container is read in the loop (set by can_be_applied)
    bool container_written_ = false; ///< Container is written in the loop (set by can_be_applied)

    /// Copy the tile in before the loop iff the container is read there.
    bool needs_copy_in() const { return container_read_; }

    /// Copy the tile back after the loop iff the container is written there.
    bool needs_copy_out() const { return container_written_; }

    /// Hard capacity guard: max scalar slots the local buffer may occupy.
    size_t max_tile_elements() const { return 1u << 16; }

    /// Private/CPU path: a nested sequential copy nest around the loop (copy-in
    /// when @p writeback is false, copy-out when true).
    void emit_private_copy(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& parent,
        const TileBuffer& buffer,
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
        structured_control_flow::Sequence& parent,
        const TileBuffer& buffer,
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
        const TileBuffer& buffer,
        const types::IType& buffer_type,
        const types::IType& pointer_type
    );

    /// Redirect every container access in the loop body to the local buffer,
    /// prefixed by the per-thread @p slot_indices.
    void rewrite_body(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const TileBuffer& buffer,
        const types::IType& buffer_type,
        const std::vector<symbolic::Expression>& slot_indices
    );

public:
    /**
     * @brief Construct a local-storage transformation for @p access_node's
     *        container within @p loop.
     *
     * The copy direction (in/out) and the storage space are both *derived* by
     * can_be_applied() from the dataflow and the enclosing parallel schedule.
     */
    LocalStorage(structured_control_flow::StructuredLoop& loop, const data_flow::AccessNode& access_node)
        : loop_(loop), access_node_(access_node), container_(access_node.data()),
          storage_type_(types::StorageType::CPU_Stack()) {}

    std::string name() const override { return "LocalStorage"; }

    /**
     * @brief Precondition check.
     *
     * Verifies the container is an existing pointer used (read and/or written)
     * in the loop, resolves a single usable tile group with provably integer
     * extents, derives the storage space from the schedule, and populates
     * tile_info_ — rejecting any schedule the apply path cannot yet handle.
     */
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Allocate the buffer, emit the derived copies, and redirect
     *        loop-body accesses to the buffer.
     */
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
