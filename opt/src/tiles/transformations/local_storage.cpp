#include "sdfg/tiles/transformations/local_storage.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <unordered_set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/atomic_op_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/tiles/analysis/tile_analysis.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/locality.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/tiles/tile_target_registry.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace transformations {

namespace {

// Assumptions for discharging a copy's boundary guard: the enclosing scope's
// assumptions (tile-tight bounds + coupled constraints on the grid/block
// indvars) plus the freshly-created copy indvars pinned to their own [0, size-1]
// range. `is_le` then proves `base + idx <= max` for a fully-covering tile.
symbolic::Assumptions build_copy_discharge_assumptions(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& scope,
    const std::vector<symbolic::Expression>& indvars,
    const std::vector<symbolic::Expression>& inclusive_uppers
) {
    auto& aa = analysis_manager.get<analysis::AssumptionsAnalysis>();
    symbolic::Assumptions assums = aa.get(scope, /*include_trivial_bounds=*/true);
    for (size_t i = 0; i < indvars.size() && i < inclusive_uppers.size(); ++i) {
        if (!SymEngine::is_a<SymEngine::Symbol>(*indvars[i]) || inclusive_uppers[i].is_null()) {
            continue;
        }
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(indvars[i]);
        symbolic::Assumption a(sym);
        a.add_lower_bound(symbolic::integer(0));
        a.tight_lower_bound(symbolic::integer(0));
        a.add_upper_bound(inclusive_uppers[i]);
        a.tight_upper_bound(inclusive_uppers[i]);
        assums.insert_or_assign(sym, a);
    }
    return assums;
}

/// Build the nested-array buffer type `elem[axes[0]][axes[1]]...` (outermost
/// first). Only the outermost array carries @p storage; the inner arrays and the
/// scalar element keep default storage so the declaration emits a single storage
/// qualifier. Each array level's num_elements is the per-axis stride source, so
/// codegen recovers multi-dimensional strides directly from the type. A
/// degenerate (no-axis) tile becomes a single-element [1] buffer.
std::unique_ptr<types::IType> make_nested_array(
    const std::vector<symbolic::Expression>& axes, const types::IType& scalar, const types::StorageType& storage
) {
    // 16-byte alignment on shared tiles lets the cooperative float4 cp.async /
    // LDS.128 addresses be provably aligned (emitted as __attribute__((aligned(16)))).
    const size_t align = storage.is_nv_shared() ? 16 : 0;
    if (axes.empty()) {
        return std::make_unique<types::Array>(storage, align, "", scalar, symbolic::integer(1));
    }
    std::unique_ptr<types::IType> inner = scalar.clone();
    for (size_t a = axes.size() - 1; a >= 1; a--) {
        inner = std::make_unique<types::Array>(*inner, axes[a]);
    }
    return std::make_unique<types::Array>(storage, align, "", *inner, axes[0]);
}

/// True if @p scope's body reads @p container in any block.
bool scope_reads_container(structured_control_flow::ControlFlowNode& scope, const std::string& container) {
    bool reads = false;
    visitor::for_each_block(scope, [&](structured_control_flow::Block& block) {
        for (auto* access : block.dataflow().data_nodes()) {
            if (access->data() == container) {
                reads = true;
            }
        }
    });
    return reads;
}

/// Block-level GPU-offloaded loops in @p loop's body that access @p container.
std::vector<structured_control_flow::StructuredLoop*> block_scheduled_consumers(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    std::vector<structured_control_flow::StructuredLoop*> consumers;
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(desc);
        if (!sl) {
            continue;
        }
        if (tiles::AxisSchedule::drives_cooperative_copy(sl->schedule_type()) &&
            scope_reads_container(sl->root(), container)) {
            consumers.push_back(sl);
        }
    }
    return consumers;
}

/// True if two tiles have the same per-dimension base and extent.
bool same_tile_shape(const analysis::MemoryTile& a, const analysis::MemoryTile& b) {
    if (a.min_subset.size() != b.min_subset.size()) {
        return false;
    }
    for (size_t d = 0; d < a.min_subset.size(); d++) {
        if (!symbolic::eq(a.min_subset[d], b.min_subset[d])) {
            return false;
        }
    }
    auto ea = a.extents_approx();
    auto eb = b.extents_approx();
    if (ea.size() != eb.size()) {
        return false;
    }
    for (size_t d = 0; d < ea.size(); d++) {
        if (ea[d].is_null() != eb[d].is_null()) {
            return false;
        }
        if (!ea[d].is_null() && !symbolic::eq(ea[d], eb[d])) {
            return false;
        }
    }
    return true;
}

/// The enclosing block-level *_Offload Map along one of @p coop_dims — the axis
/// whose threads cooperatively stage the shared tile. Unlike the immediate
/// enclosing map, this is a *genuine* cooperative axis: in a mixed
/// per-thread+cooperative tile the immediate parent is a per-thread axis, and
/// parallelizing the copy along it would stride over the slot axis and leave each
/// slot only partially filled. Returns nullptr if no cooperative dim resolves to
/// an enclosing offload Map.
structured_control_flow::Map* find_cooperative_offload_map(
    structured_control_flow::StructuredLoop& loop, const std::vector<tiles::TileAxis>& coop_dims
) {
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (!map) {
            continue;
        }
        // The cooperative copy is performed by the threads of a block, so only a
        // block-level (Group) offload schedule can drive it (a grid axis selects the
        // block; a legacy fused schedule cannot host a separate copy Map).
        if (!tiles::AxisSchedule::drives_cooperative_copy(map->schedule_type())) {
            continue;
        }
        for (const auto& d : coop_dims) {
            if (symbolic::eq(map->indvar(), d.indvar())) {
                return map;
            }
        }
    }
    return nullptr;
}

/// Resolve the tile target that owns @p loop: the innermost enclosing scheduled
/// Map. Returns nullptr when the loop is not inside any registered Map schedule.
const tiles::TileTarget* enclosing_tile_target(structured_control_flow::StructuredLoop& loop) {
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(node)) {
            if (auto* target = tiles::TileTargetRegistry::instance().get(map->schedule_type().value())) {
                return target;
            }
        }
    }
    return nullptr;
}

} // namespace

std::vector<size_t> LocalStorage::TileInfo::varying_dims() const {
    std::vector<size_t> dims;
    for (size_t d = 0; d < dimensions.size(); d++) {
        if (!symbolic::eq(dimensions.at(d), symbolic::integer(1))) {
            dims.push_back(d);
        }
    }
    return dims;
}

std::vector<symbolic::Expression> LocalStorage::TileInfo::varying_sizes() const {
    std::vector<symbolic::Expression> sizes;
    for (size_t d : varying_dims()) {
        sizes.push_back(dimensions.at(d));
    }
    return sizes;
}

tiles::Layout LocalStorage::TileInfo::source_layout() const {
    // Fold every dim's base*stride into the offset; the varying dims keep their
    // strides as the layout modes (extent-1 dims contribute only their base).
    symbolic::MultiExpression shape, stride;
    symbolic::Expression folded_offset = offset;
    for (size_t d = 0; d < dimensions.size(); d++) {
        folded_offset = symbolic::add(folded_offset, symbolic::mul(strides.at(d), bases.at(d)));
    }
    for (size_t v : varying_dims()) {
        shape.push_back(dimensions.at(v));
        stride.push_back(strides.at(v));
    }
    return tiles::Layout(shape, stride, folded_offset);
}

std::vector<symbolic::Expression> LocalStorage::TileInfo::original_subset(const std::vector<symbolic::Expression>&
                                                                              tile_indices) const {
    return {source_layout().resolve_element(tile_indices, /*require_to_element=*/false)};
}

std::vector<symbolic::Expression> LocalStorage::TileInfo::local_index(const std::vector<symbolic::Expression>&
                                                                          access_subset) const {
    std::vector<symbolic::Expression> local;
    for (size_t d = 0; d < dimensions.size(); d++) {
        if (!symbolic::eq(dimensions.at(d), symbolic::integer(1))) {
            local.push_back(symbolic::sub(access_subset.at(d), bases.at(d)));
        }
    }
    return local;
}

bool LocalStorage::has_side_effect(structured_control_flow::StructuredLoop& loop) {
    bool found = false;
    visitor::for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
        if (found) {
            return;
        }
        for (auto* lib_node : block.dataflow().library_nodes()) {
            // A __syncthreads barrier accesses no data (a control-only scheduling
            // primitive), so it cannot reference the localized container and does
            // not block staging — unlike genuine side effects (malloc/memset/…).
            if (dynamic_cast<data_flow::BarrierLocalNode*>(lib_node)) {
                continue;
            }
            // A cooperative TileCopyNode moves data through no_capture pointers
            // precisely described by pointer_access_type, so the per-container alias
            // analysis already accounts for it; it cannot independently reach the
            // localized container. Its side_effect flag only keeps DCE from dropping it.
            if (dynamic_cast<tiles::TileCopyNode*>(lib_node)) {
                continue;
            }
            if (lib_node->side_effect()) {
                found = true;
                return;
            }
        }
    });
    return found;
}

bool LocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    tile_info_ = TileInfo{};
    group_memlets_.clear();
    container_read_ = false;
    container_written_ = false;

    // Container must exist and be a pointer.
    if (!sdfg.exists(container_)) {
        return false;
    }
    if (sdfg.type(container_).type_id() != types::TypeID::Pointer) {
        return false;
    }

    // A reduction accumulator may be localized only when the owning Reduce is
    // non-cooperative (sequential / per-thread): LocalStorage privatizes the
    // accumulator and apply() retargets the Reduce's descriptor to the local
    // buffer. A cooperatively-combined (GPU-offloaded) Reduce, or one enclosing
    // the localized scope, is left to the reduce dispatcher.
    reduce_retargets_.clear();
    grid_reduce_owners_.clear();
    atomic_merge_ = false;
    if (tiles::is_reduction_accumulator(loop_, container_, analysis_manager)) {
        if (!tiles::collect_reduction_owners(loop_, container_, analysis_manager, reduce_retargets_)) {
            return false;
        }
        // A grid-parallel ancestor reduce is privatized here (per-block partial) and
        // its cross-block merge becomes an atomic copy-out; record it for apply().
        grid_reduce_owners_ = tiles::collect_grid_reduction_owners(loop_, container_, analysis_manager);
        atomic_merge_ = !grid_reduce_owners_.empty();
    }

    // Classify the container's accesses directly from the dataflow.
    auto summary = tiles::TileAnalysis::summarize(sdfg, loop_, container_);
    container_read_ = summary.reads;
    container_written_ = summary.writes;

    // Aliasing or side effects can reach the container outside the memlets we
    // rewrite, making localization unsound.
    if (summary.aliased) {
        return false;
    }
    if (has_side_effect(loop_)) {
        return false;
    }

    // Nothing to localize unless the container is actually used.
    if (!container_read_ && !container_written_) {
        return false;
    }

    // Enclosing-scope cooperative staging: a read-only tile localized at a GPU map
    // whose body has block-scheduled consumers. The tile is per-block (the map's own
    // indvar is fixed per instance), so resolve its shape from the consumers — where
    // that indvar is opaque — rather than at loop_, where it would unfold across the
    // whole grid. Stage once, reuse across every sibling consumer below.
    if (container_read_ && !container_written_) {
        tiles::LocalityPlan topo =
            tiles::LocalityPlan::analyze(loop_, tiles::TileAxis::enclosing(loop_, {}), analysis_manager);
        if (topo.enclosing_cooperative()) {
            auto consumers = block_scheduled_consumers(loop_, container_, analysis_manager);
            if (consumers.empty()) {
                return false;
            }
            const auto* target = tiles::TileTargetRegistry::instance().get(consumers.front()->schedule_type().value());
            if (!target) {
                return false;
            }
            const analysis::MemoryTileGroup* ref =
                tiles::localizable_tile(*consumers.front(), container_, analysis_manager);
            if (!ref || !tiles::is_constant_bounded(ref)) {
                return false;
            }
            auto count = tiles::tile_element_count(ref);
            auto budget = symbolic::integer(static_cast<int64_t>(max_tile_elements()));
            if (count.is_null() || !symbolic::is_true(symbolic::Le(count, budget))) {
                return false;
            }
            // Every block consumer must localize the same per-block tile; union their
            // memlets so rewrite_body repoints all of them to the shared buffer.
            group_memlets_.clear();
            for (auto* c : consumers) {
                const analysis::MemoryTileGroup* g = tiles::localizable_tile(*c, container_, analysis_manager);
                if (!g || !same_tile_shape(g->tile, ref->tile)) {
                    return false;
                }
                group_memlets_.insert(g->memlets.begin(), g->memlets.end());
            }
            auto& rt = ref->tile;
            tile_info_.dimensions = rt.extents_approx();
            tile_info_.bases = rt.min_subset;
            tile_info_.strides =
                std::vector<symbolic::Expression>(rt.layout.strides().begin(), rt.layout.strides().end());
            tile_info_.offset = rt.layout.offset();
            plan_ = topo;
            storage_type_ = target->storage_type(tiles::Space::Shared);
            return true;
        }
    }

    // Resolve the single localizable tile for the whole container.
    auto* group = tiles::localizable_tile(loop_, container_, analysis_manager);
    if (!group) {
        return false;
    }

    // Extents must be compile-time integer constants.
    if (!tiles::is_constant_bounded(group)) {
        return false;
    }

    // Physical capacity: the buffer must fit the target budget.
    auto count = tiles::tile_element_count(group);
    if (count.is_null()) {
        return false;
    }
    auto budget = symbolic::integer(static_cast<int64_t>(max_tile_elements()));
    if (!symbolic::is_true(symbolic::Le(count, budget))) {
        return false;
    }

    // Populate tile info + group memlets for apply().
    auto& t = group->tile;
    tile_info_.dimensions = t.extents_approx();
    tile_info_.bases = t.min_subset;
    tile_info_.maxes = t.max_subset;
    tile_info_.strides = std::vector<symbolic::Expression>(t.layout.strides().begin(), t.layout.strides().end());
    tile_info_.offset = t.layout.offset();
    group_memlets_.insert(group->memlets.begin(), group->memlets.end());

    // Derive the storage space from the enclosing parallel schedule.
    plan_ = tiles::LocalityPlan::analyze(loop_, tiles::TileAxis::enclosing(loop_, tile_info_.bases), analysis_manager);
    auto space = plan_.required_space(container_written_);
    if (!space) {
        return false;
    }
    switch (*space) {
        case tiles::Space::Register:
            if (const auto* target = enclosing_tile_target(loop_)) {
                storage_type_ = target->storage_type(tiles::Space::Register);
            } else {
                storage_type_ = types::StorageType::CPU_Stack();
            }
            break;
        case tiles::Space::Shared: {
            // Cooperative shared-memory path (also handles a per-thread + cooperative
            // mix, e.g. shared-memory GEMM). v1 gate: all enclosing parallel dims are
            // GPU block-level, exactly one is the cooperative (copy) axis, the tile is
            // read-only, and the cooperative Map is the loop's immediate enclosing loop.
            if (container_written_) {
                return false;
            }
            if (plan_.enclosing_cooperative()) {
                // Enclosing-scope staging: the localized GPU map's body has a
                // block-scheduled consumer supplying the copy schedule; the buffer is
                // per-block shared, loaded once and reused by every sibling below.
                auto* coop = tiles::find_block_scheduled_descendant(loop_, analysis_manager);
                if (!coop) {
                    return false;
                }
                const auto* target = tiles::TileTargetRegistry::instance().get(coop->schedule_type().value());
                if (!target) {
                    return false;
                }
                storage_type_ = target->storage_type(tiles::Space::Shared);
                break;
            }
            // v2: the tile is cooperative across >=1 block axis and may be
            // per-thread across others (each per-thread axis owns a buffer slot,
            // handled by apply()). Grid dims are permitted too: they select the
            // block (fixed per block, folded into the tile bases), so a read tile
            // that is grid-cooperative simply replicates its shared copy per block.
            // At least one cooperative axis must resolve to an enclosing block-level
            // *_Offload Map to drive the cooperative copy. The old v1 constraints
            // (exactly one cooperative axis, and it being the immediate enclosing
            // map) rejected 2D-block GEMM shared tiles.
            for (const auto& d : plan_.axes()) {
                if (!d.schedule().has_scratchpad()) {
                    return false; // no host (global-only) dims in the mix
                }
                if (d.schedule().level() != tiles::Level::Group && d.schedule().level() != tiles::Level::Device) {
                    return false; // no warp dims
                }
            }
            auto coop_dims = plan_.cooperative_axes();
            if (coop_dims.empty()) {
                return false;
            }
            auto* coop_map = find_cooperative_offload_map(loop_, coop_dims);
            if (coop_map == nullptr) {
                return false;
            }
            const auto* target = tiles::TileTargetRegistry::instance().get(coop_map->schedule_type().value());
            if (!target) {
                return false;
            }
            storage_type_ = target->storage_type(tiles::Space::Shared);
            break;
        }
        case tiles::Space::Global:
            // Grid-cooperative tiles need global memory + grid-wide sync, which is
            // not yet implemented.
            return false;
    }
    return true;
}

void LocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto* parent = dyn_cast<structured_control_flow::Sequence*>(loop_.get_parent());
    if (!parent) {
        throw InvalidTransformationException("LocalStorage: parent of loop must be a Sequence");
    }

    // Element type from a representative group memlet (container type may be opaque).
    auto* representative = *group_memlets_.begin();
    types::Scalar scalar_type(representative->base_type().primitive_type());
    types::Pointer pointer_type(scalar_type);

    local_name_ = builder.find_new_name("__daisy_local_storage_" + container_);

    // Per-thread buffer-slot prefix (mixed case): one slot dim per GPU per-thread
    // dim, sized by its block width. The within-block thread index is the map
    // indvar modulo the block width (offload maps run from 0, stride 1), which —
    // unlike a raw threadIdx symbol — is a real container the type system knows.
    std::vector<symbolic::Expression> slot_sizes;
    std::vector<symbolic::Expression> slot_indices;
    std::vector<symbolic::Expression> slot_indvars; // offload indvar per slot (flat-gather address)
    std::vector<symbolic::Expression> slot_inits;
    std::vector<symbolic::Expression> slot_strides;
    if (storage_type_.is_nv_shared()) {
        for (const auto& d : plan_.private_axes()) {
            // Only block-level per-thread dims own a buffer slot. A grid per-thread
            // dim selects the block (fixed for all its threads) and is already
            // folded into the tile bases, so it carries no slot.
            if (d.schedule().level() != tiles::Level::Group) {
                continue;
            }
            slot_sizes.push_back(d.schedule().parallel_size());
            // Within-block thread index = (indvar - init) / stride, then wrapped by
            // the block width. Tiled thread-tile offload maps are NOT normalized
            // (init = per-block base, stride = tile step), so a raw indvar % width
            // aliases distinct threads onto the same slot (e.g. stride 8, width 16
            // over a 128-wide block collapses 16 threads to 2 slots).
            symbolic::Expression tid = symbolic::div(symbolic::sub(d.indvar(), d.init()), d.stride());
            slot_indices.push_back(symbolic::mod(tid, d.schedule().parallel_size()));
            slot_indvars.push_back(d.indvar());
            slot_inits.push_back(d.init());
            slot_strides.push_back(d.stride());
        }
    }

    tiles::PackedBuffer buffer{slot_sizes, tile_info_.varying_sizes()};
    // GPU shared buffers with a per-thread slot prefix use a flat per-slot block so
    // a warp's per-slot accesses hit distinct banks: Padded (stride coprime with 32)
    // by default, or Swizzle (XOR the inner index) when opted in with a constant
    // power-of-two block. CPU/private buffers keep the dense multi-dim layout.
    if (lane_contiguous_ && storage_type_.is_nv_shared()) {
        // Fully flat, thread-linear staging for the CDNA async global->LDS DMA
        // (global_load_lds writes lane-contiguous from a wave-uniform base).
        buffer.kind = tiles::BufferKind::Linearized;
    } else if (transpose_layout_ && storage_type_.is_nv_shared() && slot_sizes.empty()) {
        // Cooperative (no-slot) tile stored column-major, so consumers read it
        // transposed. A pure affine relabelling — no padding, no per-thread slot.
        buffer.kind = tiles::BufferKind::Transposed;
    } else if (storage_type_.is_nv_shared() && !slot_sizes.empty()) {
        buffer.kind = tiles::BufferKind::Padded;
        if (swizzle_layout_) {
            auto total = buffer.tile_total_size();
            if (SymEngine::is_a<SymEngine::Integer>(*total)) {
                auto n = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
                if (n > 0 && (n & (n - 1)) == 0) {
                    buffer.kind = tiles::BufferKind::Swizzle;
                }
            }
        }
    }
    // Cooperative-store conflict avoidance: pad the inner stride to the coop axis's
    // per-warp thread count (mod 32). Compute it from the block dims + the coop
    // copy's axis (A tiles are coop over X, B over Y -> different spans, so a single
    // odd stride leaves one of them 2-way conflicted; this makes both conflict-free).
    if (buffer.kind == tiles::BufferKind::Padded) {
        if (auto* coop_map = find_cooperative_offload_map(loop_, plan_.cooperative_axes())) {
            auto block_width = [&](unsigned axis) -> size_t {
                for (const auto& d : plan_.axes()) {
                    if (d.schedule().level() == tiles::Level::Group && d.schedule().spatial_axis() == axis &&
                        SymEngine::is_a<SymEngine::Integer>(*d.schedule().parallel_size())) {
                        return static_cast<
                            size_t>(SymEngine::rcp_static_cast<const SymEngine::Integer>(d.schedule().parallel_size())
                                        ->as_int());
                    }
                }
                return 1;
            };
            const auto* warp_target = tiles::TileTargetRegistry::instance().get(coop_map->schedule_type().value());
            const size_t warp = warp_target ? static_cast<size_t>(warp_target->lane_width()) : 1;
            const size_t bx = block_width(/*X=*/0);
            const size_t by = block_width(/*Y=*/1);
            const size_t bz = block_width(/*Z=*/2);
            // Per-warp thread counts, x fastest in the flat thread index.
            const size_t x_per = std::min(bx, warp);
            const size_t rem_y = std::max<size_t>(1, warp / std::max<size_t>(1, x_per));
            const size_t y_per = std::min(by, rem_y);
            const size_t rem_z = std::max<size_t>(1, rem_y / std::max<size_t>(1, y_per));
            const size_t z_per = std::min(bz, rem_z);
            // The cooperative copy's own spatial axis (0=X, 1=Y, 2=Z) selects which
            // per-warp span pads the inner stride.
            auto coop_schedule = tiles::AxisSchedule::classify(coop_map->schedule_type());
            switch (coop_schedule ? coop_schedule->spatial_axis() : 0) {
                case 0:
                    buffer.coop_warp_span = x_per;
                    break;
                case 1:
                    buffer.coop_warp_span = y_per;
                    break;
                case 2:
                    buffer.coop_warp_span = z_per;
                    break;
                default:
                    break;
            }
        }
    }
    // Multi-dimensional (nested-array) buffer: one array level per [slot ++ tile]
    // axis, so every access is a clean per-axis subset and clang recovers the
    // strides from each level's num_elements (instead of a single collapsed
    // linear index with div/mod that defeats load/store vectorization).
    auto buffer_type_ptr = make_nested_array(buffer.axes(), scalar_type, storage_type_);
    auto& buffer_type = *buffer_type_ptr;
    builder.add_container(local_name_, buffer_type);

    if (storage_type_.is_nv_shared()) {
        if (plan_.enclosing_cooperative()) {
            // Stage the row once at the top of the localized GPU map's body; the
            // (sibling) block consumers below read the shared buffer.
            auto* coop = tiles::find_block_scheduled_descendant(loop_, analysis_manager);
            auto impl = tiles::TileTargetRegistry::instance().implementation_type(coop->schedule_type().value());
            auto& body = loop_.root();
            auto copy = build_tiled_copy(
                analysis_manager,
                body,
                buffer,
                pointer_type,
                slot_sizes,
                slot_indices,
                slot_indvars,
                slot_inits,
                slot_strides
            );
            emit_copy_node(
                builder,
                body,
                body.at(0),
                /*after=*/false,
                copy,
                impl,
                pointer_type,
                tiles::CopyDirection::In,
                /*leading_barrier=*/false,
                /*trailing_barrier=*/true
            );
        } else {
            // Cooperative copy-in before the loop. A per-thread slot prefix re-stages
            // the tile each coverage iteration, so a leading barrier guards the
            // overwrite against the previous iteration's outstanding reads.
            auto* coop_map = find_cooperative_offload_map(loop_, plan_.cooperative_axes());
            auto impl = tiles::TileTargetRegistry::instance().implementation_type(coop_map->schedule_type().value());
            auto copy = build_tiled_copy(
                analysis_manager,
                *parent,
                buffer,
                pointer_type,
                slot_sizes,
                slot_indices,
                slot_indvars,
                slot_inits,
                slot_strides
            );
            emit_copy_node(
                builder,
                *parent,
                loop_,
                /*after=*/false,
                copy,
                impl,
                pointer_type,
                tiles::CopyDirection::In,
                /*leading_barrier=*/!slot_indices.empty(),
                /*trailing_barrier=*/true
            );
        }
    } else {
        if (needs_copy_in()) {
            emit_private_copy(builder, analysis_manager, *parent, buffer, buffer_type, pointer_type, /*writeback=*/false);
        }
        if (needs_copy_out()) {
            emit_private_copy(builder, analysis_manager, *parent, buffer, buffer_type, pointer_type, /*writeback=*/true);
        }
    }

    rewrite_body(builder, analysis_manager, buffer, buffer_type, slot_indices);

    // Privatized reduction accumulator: point each owning (non-cooperative) Reduce
    // at the local buffer so its denormalized descriptor matches the rewritten
    // dataflow. The copy-in seeds it from the original and the copy-out stores back.
    for (auto* reduce : reduce_retargets_) {
        reduce->replace_reduction_container(container_, local_name_);
    }

    // Grid-parallel reductions: the atomic copy-out now performs the cross-block
    // merge, so demote each owning Reduce to a plain Map (the reduce dispatcher no
    // longer handles this accumulator).
    for (auto* reduce : grid_reduce_owners_) {
        if (auto* parent = dynamic_cast<structured_control_flow::Sequence*>(reduce->get_parent())) {
            builder.convert_reduce_to_map(*parent, *reduce);
        }
    }

    analysis_manager.invalidate_all();
}

symbolic::Condition LocalStorage::boundary_guard(
    const data_flow::Subset& tile_indices, const symbolic::SymbolSet& params, const symbolic::Assumptions& assums
) const {
    // Compare each delinearized global index (base[d] + tile_index) against the
    // tile's max valid index maxes[d]. tile_indices are per varying dim, aligned
    // with varying_dims(); degenerate (extent-1) dims sit at their base <= max.
    // A conjunct provably always true under the assumptions (a fully-covering
    // tile: `base + idx <= max` holds for every idx) is dropped — sound, and it
    // lets an interior copy vectorize instead of sitting under a predicate.
    symbolic::Condition guard = SymEngine::boolTrue;
    auto vdims = tile_info_.varying_dims();
    auto vsizes = tile_info_.varying_sizes();
    for (size_t v = 0; v < vdims.size() && v < tile_indices.size(); ++v) {
        size_t d = vdims[v];
        if (d >= tile_info_.maxes.size() || tile_info_.maxes[d].is_null()) {
            continue;
        }
        auto global_d = symbolic::add(tile_info_.bases[d], tile_indices[v]);
        // Discharge against the worst-case (fully-covering) coordinate `size-1`
        // rather than the actual index expression: the copy sweeps [0, size-1] and
        // maxes[d] is index-independent, so `base + (size-1) <= max` implies
        // `base + idx <= max` for every idx. This is robust when tile_indices[v] is
        // a non-symbol idiv/imod of a flat coverage indvar (build_copy_discharge
        // only pins plain symbols, so such coordinates are otherwise unbounded and
        // the interior guard never discharges).
        symbolic::Expression probe = global_d;
        if (v < vsizes.size() && !vsizes[v].is_null()) {
            probe = symbolic::add(tile_info_.bases[d], symbolic::sub(vsizes[v], symbolic::integer(1)));
        }
        if (!assums.empty() && symbolic::is_le(probe, tile_info_.maxes[d], params, assums, /*tight=*/true)) {
            continue;
        }
        guard = symbolic::And(guard, symbolic::Le(global_d, tile_info_.maxes[d]));
    }
    return guard;
}

void LocalStorage::emit_private_copy(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& parent,
    const tiles::PackedBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type,
    bool writeback
) {
    if (!atomic_merge_) {
        // Plain copy: the unified dense whole-block TileCopyNode (its reference
        // dispatcher emits a vectorizable per-mode loop). Copy-in stages before the
        // loop, copy-out writes back after it; no barriers on the private path.
        auto copy = build_tiled_copy(analysis_manager, parent, buffer, pointer_type, {}, {}, {}, {}, {});
        emit_copy_node(
            builder,
            parent,
            loop_,
            /*after=*/writeback,
            copy,
            data_flow::ImplementationType_NONE,
            pointer_type,
            writeback ? tiles::CopyDirection::Out : tiles::CopyDirection::In,
            /*leading_barrier=*/false,
            /*trailing_barrier=*/false
        );
        return;
    }

    // Atomic-merge path: a per-element gather with atomic accumulation (or a
    // zero-init) that the tile copy node does not yet express, so build it directly.
    auto varying_dims = tile_info_.varying_dims();
    auto varying_dim_sizes = tile_info_.varying_sizes();
    int index = parent.index(loop_) + (writeback ? 1 : 0);
    auto& scope = writeback ? builder.add_sequence_after(parent, loop_, loop_.debug_info())
                            : builder.add_sequence_before(parent, loop_, loop_.debug_info());

    // Element-predicate the global access: the over-approximated tile may address
    // out-of-bounds global memory on ragged blocks. Skip those elements (their buffer
    // slots are never consumed). Provably in-bounds conjuncts (a fully-covering tile)
    // are dropped so the interior copy vectorizes.
    std::vector<symbolic::Expression> incl_uppers;
    for (const auto& s : varying_dim_sizes) {
        incl_uppers
            .push_back(s.is_null() ? symbolic::Expression(SymEngine::null) : symbolic::sub(s, symbolic::integer(1)));
    }
    auto params = analysis_manager.get<analysis::AssumptionsAnalysis>().parameters();
    auto guard_of = [&](const std::vector<symbolic::Expression>& idx) {
        auto discharge = build_copy_discharge_assumptions(analysis_manager, parent, idx, incl_uppers);
        return boundary_guard(idx, params, discharge);
    };

    structured_control_flow::Sequence* current = &scope;
    std::vector<symbolic::Expression> indvars;
    for (size_t i = 0; i < varying_dims.size(); i++) {
        auto name = builder.find_new_name(
            "__daisy_ls_" + std::string(writeback ? "wb" : "ci") + "_" + container_ + "_d" +
            std::to_string(varying_dims[i])
        );
        // Int32: this copy index sweeps [0, tile_dim), a compile-time-constant tile
        // extent (can_be_applied enforces is_constant_bounded + max_tile_elements),
        // so it always fits int32. In the global address it is added to a 64-bit
        // base, so C promotion keeps that arithmetic 64-bit safe for any N.
        builder.add_container(name, types::Scalar(types::PrimitiveType::Int32));
        auto indvar = symbolic::symbol(name);
        indvars.push_back(indvar);
        auto& map = builder.add_map(
            *current,
            indvar,
            symbolic::Lt(indvar, varying_dim_sizes[i]),
            symbolic::integer(0),
            symbolic::add(indvar, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create(),
            loop_.debug_info()
        );
        current = &map.root();
    }

    data_flow::Subset original_subset = tile_info_.original_subset(indvars);
    data_flow::Subset buffer_subset = buffer.subset({}, indvars);
    auto guard = guard_of(indvars);
    structured_control_flow::Sequence* body = current;
    if (!symbolic::is_true(guard)) {
        auto& if_else = builder.add_if_else(*current, loop_.debug_info());
        body = &builder.add_case(if_else, guard, loop_.debug_info());
    }

    // Scalar element type of the accumulator (the container pointer's pointee).
    const types::IType* scalar_type = &buffer_type;
    if (auto* p = dynamic_cast<const types::Pointer*>(&pointer_type)) {
        if (p->has_pointee_type()) {
            scalar_type = &p->pointee_type();
        }
    }

    if (writeback) {
        // Atomically merge this block's partial into the global accumulator. Both
        // edges into the atomic node carry no subset: stage the tile element into a
        // scalar, and pre-offset the global slot via a reference memlet.
        const std::string sched_val = grid_reduce_owners_.front()->schedule_type().value();
        const data_flow::AtomicScalarOpImpl* impl;
        if (sched_val == "CUDA_Offload") {
            impl = data_flow::AtomicScalarOpCudaImpl::instance();
        } else if (sched_val == "ROCM_Offload") {
            impl = data_flow::AtomicScalarOpRocmImpl::instance();
        } else {
            impl = data_flow::AtomicScalarOpCPUImpl::instance();
        }

        if (!impl->supports(scalar_type->primitive_type(), data_flow::AtomicOpType::Add)) {
            throw InvalidTransformationException(
                "LocalStorage: atomic merge of type " +
                std::string(types::primitive_type_to_string(scalar_type->primitive_type())) +
                " not supported on impl " + std::string(impl->type_name())
            );
        }

        auto val_name = builder.find_new_name("__daisy_atom_val_" + container_);
        builder.add_container(val_name, *scalar_type);
        auto& b1 = builder.add_block(*body);
        auto& tile_src = builder.add_access(b1, local_name_);
        auto& val_w = builder.add_access(b1, val_name);
        auto& stage = builder.add_tasklet(b1, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(b1, tile_src, stage, "_in", buffer_subset, buffer_type);
        builder.add_computational_memlet(b1, stage, "_out", val_w, {}, *scalar_type);

        auto ptr_name = builder.find_new_name("__daisy_atom_ptr_" + container_);
        builder.add_container(ptr_name, pointer_type);
        auto& b2 = builder.add_block(*body);
        auto& acc_src = builder.add_access(b2, container_);
        auto& dref_w = builder.add_access(b2, ptr_name);
        builder.add_reference_memlet(b2, acc_src, dref_w, original_subset, pointer_type);

        auto& b3 = builder.add_block(*body);
        auto& dref_r = builder.add_access(b3, ptr_name);
        auto& val_r = builder.add_access(b3, val_name);
        auto& node = builder.add_library_node<data_flow::AtomicScalarOpNode>(
            b3, loop_.debug_info(), scalar_type->primitive_type(), data_flow::AtomicOpType::Add, impl
        );
        builder.add_computational_memlet(b3, dref_r, node, "_dst", {}, pointer_type);
        builder.add_computational_memlet(b3, val_r, node, "_src", {}, *scalar_type);
    } else {
        // Zero-init the per-block partial (it accumulates only this block's k-slice).
        auto& block = builder.add_block(*body);
        auto& zero = builder.add_constant(block, "0", *scalar_type);
        auto& dst = builder.add_access(block, local_name_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {}, *scalar_type);
        builder.add_computational_memlet(block, tasklet, "_out", dst, buffer_subset, buffer_type);
    }

    builder.move_children(scope, parent, index + 1);
    builder.remove_child(parent, index);
}

tiles::TileGuard LocalStorage::tile_boundary_guard(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& guard_scope,
    const std::vector<symbolic::Expression>& vsizes
) {
    // One boundary dim per varying tile axis: `base + tile_coord <= max`. The
    // discharge (against the scope assumptions) drops dims whose worst-case
    // coordinate `base + size - 1` is provably in bounds, so a fully-covering tile
    // carries no guard. A later pass can re-discharge more dims via normalize_guard.
    tiles::TileGuard guard;
    guard.tile_sizes = vsizes;
    auto vdims = tile_info_.varying_dims();
    for (size_t v = 0; v < vdims.size() && v < vsizes.size(); ++v) {
        size_t d = vdims[v];
        if (d >= tile_info_.maxes.size() || tile_info_.maxes[d].is_null()) {
            continue;
        }
        guard.dims.push_back({v, tile_info_.bases[d], tile_info_.maxes[d]});
    }
    auto params = analysis_manager.get<analysis::AssumptionsAnalysis>().parameters();
    auto assums = build_copy_discharge_assumptions(analysis_manager, guard_scope, {}, {});
    guard.discharge(params, assums);
    return guard;
}

LocalStorage::BuiltCopy LocalStorage::build_tiled_copy(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& guard_scope,
    const tiles::PackedBuffer& buffer,
    const types::IType& pointer_type,
    const std::vector<symbolic::Expression>& slot_sizes,
    const std::vector<symbolic::Expression>& slot_indices,
    const std::vector<symbolic::Expression>& slot_indvars,
    const std::vector<symbolic::Expression>& slot_inits,
    const std::vector<symbolic::Expression>& slot_strides
) {
    BuiltCopy out;
    auto& plan = out.plan;
    auto vsizes = tile_info_.varying_sizes();
    plan.atom = tiles::CopyAtom::ScalarSync;

    if (buffer.kind == tiles::BufferKind::Linearized) {
        // Lane-contiguous flat staging (CDNA async global->LDS). Fold the slot
        // indvars (threadIdx) out of the source offset into leading affine dims, so
        // the whole block sweeps [slot ++ tile] thread-linearly into a flat buffer;
        // the node references no per-thread symbol and codegen emits an unguarded
        // full-block sweep. coop_axes empty => whole block; guard subsumed by the
        // flat bound.
        auto src_tile = tile_info_.source_layout();
        symbolic::ExpressionMapping to_init;
        for (size_t s = 0; s < slot_indvars.size(); ++s) {
            to_init[slot_indvars.at(s)] = slot_inits.at(s);
        }
        auto offset_folded = symbolic::subs(src_tile.offset(), to_init);
        std::vector<symbolic::Expression> slot_folded_strides;
        for (size_t s = 0; s < slot_indvars.size(); ++s) {
            symbolic::ExpressionMapping advance = to_init;
            advance[slot_indvars.at(s)] = symbolic::add(slot_inits.at(s), slot_strides.at(s));
            slot_folded_strides.push_back(symbolic::sub(symbolic::subs(src_tile.offset(), advance), offset_folded));
        }
        symbolic::MultiExpression src_shape = slot_sizes;
        src_shape.insert(src_shape.end(), vsizes.begin(), vsizes.end());
        symbolic::MultiExpression src_stride = slot_folded_strides;
        src_stride.insert(src_stride.end(), src_tile.strides().begin(), src_tile.strides().end());
        symbolic::MultiExpression dst_stride(src_shape.size());
        symbolic::Expression run = symbolic::integer(1);
        for (int d = static_cast<int>(src_shape.size()) - 1; d >= 0; --d) {
            dst_stride[d] = run;
            run = symbolic::mul(run, src_shape[d]);
        }
        plan.src = tiles::Layout(src_shape, src_stride, offset_folded);
        plan.dst = tiles::Layout(src_shape, dst_stride, symbolic::integer(0));
        // Whole-block flat sweep bounded by the flat size — no per-element guard.
        out.guard = {};
    } else if (!slot_indices.empty()) {
        // Per-thread-slot (Padded/Swizzle): each thread stages its own slot (source
        // uses its own thread index; the slot prefix folds into the buffer offset)
        // while the cooperating Group spatial axes split the tile. Referencing the
        // slot index but not the coop index lets codegen guard by `i < N` only, so
        // every coop thread participates even when the coop axis is ragged. A Swizzle
        // buffer XORs the (slot ++ tile) offset, composed on plan.dst via dst_swizzle.
        plan.src = tile_info_.source_layout();
        auto full_dst = buffer.layout().layout;
        std::vector<symbolic::Expression> prefix_coords = slot_indices;
        for (size_t d = 0; d < vsizes.size(); ++d) {
            prefix_coords.push_back(symbolic::integer(0));
        }
        auto slot_prefix = full_dst.resolve_element(prefix_coords, /*require_to_element=*/false);
        std::vector<symbolic::Expression>
            tile_strides(full_dst.strides().begin() + slot_indices.size(), full_dst.strides().end());
        plan.dst = tiles::Layout(vsizes, tile_strides, slot_prefix);
        plan.dst_swizzle = buffer.layout().swizzle;
        for (const auto& d : plan_.cooperative_axes()) {
            if (d.schedule().level() == tiles::Level::Group) {
                out.coop_axes.push_back(static_cast<int>(d.schedule().spatial_axis()));
            }
        }
        std::sort(out.coop_axes.begin(), out.coop_axes.end());
        out.guard = tile_boundary_guard(analysis_manager, guard_scope, vsizes);
    } else {
        // Dense whole-block (MultiDim or Transposed, no slots): the buffer's affine
        // layout is the physical tile placement (row- or column-major); every block
        // thread strides the flat tile.
        plan.src = tile_info_.source_layout();
        plan.dst = buffer.layout().layout;
        plan.dst_swizzle = buffer.layout().swizzle;
        out.guard = tile_boundary_guard(analysis_manager, guard_scope, vsizes);
    }

    return out;
}

void LocalStorage::emit_copy_node(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    structured_control_flow::ControlFlowNode& anchor,
    bool after,
    const BuiltCopy& copy,
    const data_flow::ImplementationType& impl,
    const types::IType& pointer_type,
    tiles::CopyDirection direction,
    bool leading_barrier,
    bool trailing_barrier
) {
    // Insert each new block adjacent to the anchor; consecutive before-inserts stack
    // in call order (each lands just before the anchor, pushing prior ones up).
    auto add_block = [&]() -> structured_control_flow::Block& {
        return after ? builder.add_block_after(scope, anchor, loop_.debug_info())
                     : builder.add_block_before(scope, anchor, loop_.debug_info());
    };

    if (leading_barrier) {
        builder.add_library_node<data_flow::BarrierLocalNode>(add_block(), DebugInfo());
    }

    const bool copy_in = direction == tiles::CopyDirection::In;
    auto& copy_block = add_block();
    // _dst is the write target (In: buffer, Out: global); _src the read source.
    auto& dst_acc = builder.add_access(copy_block, copy_in ? local_name_ : container_);
    auto& src_acc = builder.add_access(copy_block, copy_in ? container_ : local_name_);
    const size_t bytes = types::bit_width(pointer_type.primitive_type()) / 8;
    auto& node = builder.add_library_node<tiles::TileCopyNode>(
        copy_block, loop_.debug_info(), impl, copy.plan, direction, bytes, copy.guard, copy.coop_axes
    );
    builder.add_computational_memlet(copy_block, dst_acc, node, "_dst", {}, pointer_type);
    builder.add_computational_memlet(copy_block, src_acc, node, "_src", {}, pointer_type);

    if (trailing_barrier) {
        builder.add_library_node<data_flow::BarrierLocalNode>(add_block(), DebugInfo());
    }
}

void LocalStorage::rewrite_body(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const tiles::PackedBuffer& buffer,
    const types::IType& buffer_type,
    const std::vector<symbolic::Expression>& slot_indices
) {
    // v1 guarantees single-group full coverage, so every group memlet is rewritten
    // and its access node renamed (no split-node handling needed).
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    visitor::for_each_block(loop_.root(), [&](structured_control_flow::Block& block) {
        auto& dfg = block.dataflow();
        std::vector<data_flow::AccessNode*> access_nodes;
        for (auto* access_node : dfg.data_nodes()) {
            if (access_node->data() == container_) {
                access_nodes.push_back(access_node);
            }
        }
        for (auto* access : access_nodes) {
            bool rewrote = false;
            auto rewrite_edge = [&](data_flow::Memlet& memlet) {
                if (group_memlets_.count(&memlet) == 0) {
                    return;
                }
                auto* acc = mla.access(memlet);
                if (!acc || acc->subset.size() != tile_info_.dimensions.size()) {
                    return;
                }
                memlet.set_subset(buffer.subset(slot_indices, tile_info_.local_index(acc->subset)));
                memlet.set_base_type(buffer_type);
                rewrote = true;
            };
            for (auto& memlet : dfg.out_edges(*access)) {
                rewrite_edge(memlet);
            }
            for (auto& memlet : dfg.in_edges(*access)) {
                rewrite_edge(memlet);
            }
            if (rewrote) {
                access->data(local_name_);
            }
        }
    });
}


void LocalStorage::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer serializer_full;
    j["parameters"]["storage_type"] = nlohmann::json::object();
    serializer_full.storage_type_to_json(j["parameters"]["storage_type"], storage_type_);
    j["parameters"]["swizzle_layout"] = swizzle_layout_;
    j["parameters"]["lane_contiguous"] = lane_contiguous_;
    j["parameters"]["transpose_layout"] = transpose_layout_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);

    j["subgraph"]["1"] = nlohmann::json::object();
    j["subgraph"]["1"]["element_id"] = access_node_.element_id();
    j["subgraph"]["1"]["type"] = "access_node";
}

LocalStorage LocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }

    auto access_node = dynamic_cast<
        data_flow::AccessNode*>(builder.find_element_by_id(desc.at("subgraph").at("1").at("element_id").get<size_t>()));
    if (!access_node) {
        throw InvalidTransformationDescriptionException(
            "Access node with ID " + std::to_string(desc.at("subgraph").at("1").at("element_id").get<size_t>()) +
            " not found."
        );
    }

    bool swizzle_layout = false;
    if (desc.contains("parameters") && desc["parameters"].contains("swizzle_layout")) {
        swizzle_layout = desc["parameters"]["swizzle_layout"].get<bool>();
    }

    bool lane_contiguous = false;
    if (desc.contains("parameters") && desc["parameters"].contains("lane_contiguous")) {
        lane_contiguous = desc["parameters"]["lane_contiguous"].get<bool>();
    }

    bool transpose_layout = false;
    if (desc.contains("parameters") && desc["parameters"].contains("transpose_layout")) {
        transpose_layout = desc["parameters"]["transpose_layout"].get<bool>();
    }

    return LocalStorage(*loop, *access_node, swizzle_layout, lane_contiguous, transpose_layout);
}

} // namespace transformations
} // namespace sdfg
