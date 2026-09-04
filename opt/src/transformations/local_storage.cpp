#include "sdfg/transformations/local_storage.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <unordered_set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/atomic_op_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

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

/// Visit every Block reachable under @p node (recursing through sequences,
/// loops, and if-else branches).
void for_each_block(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::Block&)>& fn
) {
    if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
        fn(*block);
    } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            for_each_block(seq->at(i), fn);
        }
    } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        for_each_block(loop->root(), fn);
    } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            for_each_block(if_else->at(i).first, fn);
        }
    }
}

/// Escape/overwrite/read/write policy for a single container, fed by the shared
/// pointer analyzers.
struct ContainerAccessPolicy {
    std::string container;
    bool reads = false;
    bool writes = false;
    bool aliased = false; ///< escaped, overwritten, or captured

    void on_escape(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_overwrite(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_read_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) reads = true;
    }
    void on_write_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) writes = true;
    }
};

/// Composes the shared PointerEscape/Overwrite/Used analyzers over a subtree,
/// mirroring MemoryOwnershipAnalysis. Adds one refinement DataDependencyAnalysis
/// carries but the analyzers do not: a library node consuming the pointer with a
/// missing or non-`no_capture` `pointer_access_type` is treated as aliasing.
class ContainerAccessVisitor : public analysis::BaseUserVisitor,
                               public analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerUsedAnalyzer<ContainerAccessPolicy> {
    ContainerAccessPolicy& policy_;

    void capture_check(const data_flow::Memlet& edge, const data_flow::DataFlowNode& other) {
        if (auto* lib = dynamic_cast<const data_flow::LibraryNode*>(&other)) {
            auto access = lib->pointer_access_type(edge);
            if (!access || !access->no_capture()) {
                policy_.aliased = true;
            }
        }
    }

public:
    ContainerAccessVisitor(const StructuredSDFG& sdfg, ContainerAccessPolicy& policy)
        : analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerUsedAnalyzer<ContainerAccessPolicy>(sdfg, policy), policy_(policy) {}

    void use_as_src_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.dst());
    }
    void use_as_dst_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.src());
    }
    void use_as_return_src(const std::string& c, const structured_control_flow::Return& r) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_return_src(c, r);
    }
    void use_as_symbol_read(
        const std::string& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_symbol_read(c, n, u, loc, loc_index, expr);
    }
    void use_as_symbol_write(
        const symbolic::Symbol& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolWriteLocation loc
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_symbol_write(c, n, u, loc);
    }
};

/// First block-level GPU-offloaded loop in @p loop's body (a cooperative copy /
/// consumer axis for enclosing-scope staging), or nullptr.
structured_control_flow::StructuredLoop* find_block_scheduled_descendant(
    structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(desc);
        if (!sl) {
            continue;
        }
        auto& sched = sl->schedule_type();
        if (sched.category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (gpu::is_block_level(gpu::gpu_target_level(sched))) {
            return sl;
        }
    }
    return nullptr;
}

/// True if @p scope's body reads @p container in any block.
bool scope_reads_container(structured_control_flow::ControlFlowNode& scope, const std::string& container) {
    bool reads = false;
    for_each_block(scope, [&](structured_control_flow::Block& block) {
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
        auto& sched = sl->schedule_type();
        if (sched.category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (gpu::is_block_level(gpu::gpu_target_level(sched)) && scope_reads_container(sl->root(), container)) {
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
    structured_control_flow::StructuredLoop& loop, const std::vector<LocalStorage::LocalityPlan::Dim>& coop_dims
) {
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (!map) {
            continue;
        }
        const std::string& sched_value = map->schedule_type().value();
        if (sched_value != "CUDA_Offload" && sched_value != "ROCM_Offload") {
            continue;
        }
        // The cooperative copy is performed by the threads of a block, so only a
        // block-level offload axis can drive it (a grid axis selects the block).
        if (!gpu::is_block_level(gpu::gpu_target_level(map->schedule_type()))) {
            continue;
        }
        for (const auto& d : coop_dims) {
            if (symbolic::eq(map->indvar(), d.indvar)) {
                return map;
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

std::vector<symbolic::Expression> LocalStorage::TileInfo::original_subset(const std::vector<symbolic::Expression>&
                                                                              tile_indices) const {
    std::vector<symbolic::Expression> full;
    size_t v = 0;
    for (size_t d = 0; d < dimensions.size(); d++) {
        if (!symbolic::eq(dimensions.at(d), symbolic::integer(1))) {
            full.push_back(symbolic::add(bases.at(d), tile_indices.at(v++)));
        } else {
            full.push_back(bases.at(d));
        }
    }
    symbolic::Expression linear = offset;
    for (size_t d = 0; d < full.size(); d++) {
        linear = symbolic::add(linear, symbolic::mul(strides.at(d), full.at(d)));
    }
    return {linear};
}

std::vector<symbolic::Expression> LocalStorage::TileInfo::flat_original_subset(
    const std::vector<symbolic::Expression>& slot_indvars,
    const std::vector<symbolic::Expression>& slot_values,
    const std::vector<symbolic::Expression>& tile_indices
) const {
    // The gather address for a flat element: original_subset with each slot indvar
    // replaced by its value for the delinearized slot index (init + stride*idx).
    auto lin = original_subset(tile_indices).at(0);
    for (size_t s = 0; s < slot_indvars.size(); s++) {
        lin = symbolic::subs(lin, slot_indvars.at(s), slot_values.at(s));
    }
    return {lin};
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

symbolic::Expression LocalStorage::TileBuffer::total_size() const {
    symbolic::Expression total = symbolic::integer(1);
    for (const auto& s : slot_sizes) {
        total = symbolic::mul(total, s);
    }
    for (const auto& s : tile_sizes) {
        total = symbolic::mul(total, s);
    }
    return total;
}

symbolic::Expression LocalStorage::TileBuffer::tile_total_size() const {
    symbolic::Expression total = symbolic::integer(1);
    for (const auto& s : tile_sizes) {
        total = symbolic::mul(total, s);
    }
    return total;
}

symbolic::Expression LocalStorage::TileBuffer::slot_offset(const std::vector<symbolic::Expression>& slot_indices
) const {
    // Row-major over slot_sizes, with the innermost slot stride = tile_total_size
    // (each slot owns a contiguous tile block).
    symbolic::Expression linear = symbolic::integer(0);
    symbolic::Expression stride = tile_total_size();
    for (int i = static_cast<int>(slot_indices.size()) - 1; i >= 0; i--) {
        linear = symbolic::add(linear, symbolic::mul(slot_indices[i], stride));
        stride = symbolic::mul(stride, slot_sizes[i]);
    }
    return linear;
}

symbolic::Expression LocalStorage::TileBuffer::linearize(
    const std::vector<symbolic::Expression>& slot_indices, const std::vector<symbolic::Expression>& tile_indices
) const {
    // Row-major over the concatenation [slot dims ++ tile dims].
    std::vector<symbolic::Expression> sizes = slot_sizes;
    sizes.insert(sizes.end(), tile_sizes.begin(), tile_sizes.end());
    std::vector<symbolic::Expression> indices = slot_indices;
    indices.insert(indices.end(), tile_indices.begin(), tile_indices.end());

    symbolic::Expression linear = symbolic::integer(0);
    symbolic::Expression stride = symbolic::integer(1);
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; i--) {
        linear = symbolic::add(linear, symbolic::mul(indices[i], stride));
        stride = symbolic::mul(stride, sizes[i]);
    }
    return linear;
}

std::vector<symbolic::Expression> LocalStorage::TileBuffer::axes() const {
    std::vector<symbolic::Expression> out = slot_sizes;
    switch (layout) {
        case Layout::Padded:
            // [slot dims][padded per-slot block]: one flat dim whose stride
            // (inner_stride) is coprime with 32.
            out.push_back(inner_stride());
            break;
        case Layout::Linearized:
            // Fully flat [total_size]: slots folded into one linear dimension
            // (lane-contiguous thread-linear staging).
            return {total_size()};
        case Layout::Swizzle:
            // [slot dims][natural per-slot block]: flat, unpadded; the XOR swizzle
            // in multi_subset spreads banks.
            out.push_back(tile_total_size());
            break;
        case Layout::MultiDim:
            out.insert(out.end(), tile_sizes.begin(), tile_sizes.end());
            break;
    }
    return out;
}

symbolic::Expression LocalStorage::TileBuffer::slot_linearize(const std::vector<symbolic::Expression>& slot_indices
) const {
    symbolic::Expression linear = symbolic::integer(0);
    symbolic::Expression stride = symbolic::integer(1);
    for (int i = static_cast<int>(slot_indices.size()) - 1; i >= 0; i--) {
        linear = symbolic::add(linear, symbolic::mul(slot_indices[i], stride));
        if (i < static_cast<int>(slot_sizes.size())) {
            stride = symbolic::mul(stride, slot_sizes[i]);
        }
    }
    return linear;
}

data_flow::Subset LocalStorage::TileBuffer::multi_subset(
    const std::vector<symbolic::Expression>& slot_indices, const std::vector<symbolic::Expression>& tile_indices
) const {
    data_flow::Subset subset = slot_indices;
    switch (layout) {
        case Layout::Linearized:
            // Fully flat: one linear index over [slot ++ tile] (slots folded).
            return {linearize(slot_indices, tile_indices)};
        case Layout::Padded:
            // Flatten the tile indices into the single flat inner dimension.
            subset.push_back(tile_linearize(tile_indices));
            break;
        case Layout::Swizzle:
            // Flatten, then XOR-swizzle by the slot so per-slot accesses hit distinct
            // banks. A bijection on the power-of-two inner range (applied identically
            // to copy-in and reads, so it is a pure relabelling of where data lives).
            subset.push_back(symbolic::bit_xor(tile_linearize(tile_indices), slot_linearize(slot_indices)));
            break;
        case Layout::MultiDim:
            subset.insert(subset.end(), tile_indices.begin(), tile_indices.end());
            break;
    }
    // A degenerate (all extent-1, no-slot) tile has no axes; the buffer is a
    // single [1] element addressed at index 0.
    if (subset.empty()) {
        subset.push_back(symbolic::integer(0));
    }
    return subset;
}

symbolic::Expression LocalStorage::TileBuffer::tile_linearize(const std::vector<symbolic::Expression>& tile_indices
) const {
    symbolic::Expression linear = symbolic::integer(0);
    symbolic::Expression stride = symbolic::integer(1);
    for (int i = static_cast<int>(tile_indices.size()) - 1; i >= 0; i--) {
        linear = symbolic::add(linear, symbolic::mul(tile_indices[i], stride));
        if (i < static_cast<int>(tile_sizes.size())) {
            stride = symbolic::mul(stride, tile_sizes[i]);
        }
    }
    return linear;
}

symbolic::Expression LocalStorage::TileBuffer::inner_stride() const {
    auto total = tile_total_size();
    // Pad only a constant block to the next odd size (coprime with 32); a
    // symbolic block cannot be safely padded, so leave it unpadded.
    if (!SymEngine::is_a<SymEngine::Integer>(*total)) {
        return total;
    }
    auto n = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
    // Cooperative-store conflict avoidance: pad the per-slot stride to be congruent
    // to the coop warp span (mod 32). A warp writes `[slot][coop]`; making adjacent
    // slots exactly `coop_warp_span` banks apart tiles the per-slot coop ranges onto
    // distinct banks for both slot-fast and slot-slow tiles. Loads only vary the
    // slot (broadcasts), so any non-zero-mod-32 stride keeps them conflict-free.
    if (coop_warp_span > 0) {
        long target = static_cast<long>(coop_warp_span % 32);
        long add = ((target - (n % 32)) % 32 + 32) % 32;
        // Round the padded stride up to a multiple of 4 (16 bytes) so a whole
        // shared row is float4-addressable (cooperative cp.async / LDS.128). When
        // coop_warp_span is itself a multiple of 4 (fast coop axis) this is a no-op
        // and stays conflict-free; a slow coop axis trades the exact bank spread for
        // 16-byte rows, which the vectorized cooperative copy more than recovers.
        long padded = (n + add + 3) & ~3L;
        return symbolic::integer(padded);
    }
    if (n % 2 == 0) {
        return symbolic::integer(n + 1);
    }
    return total;
}

std::vector<symbolic::Expression> LocalStorage::TileBuffer::delinearize_tile(const symbolic::Expression& flat) const {
    std::vector<symbolic::Expression> decomp;
    symbolic::Expression remainder = flat;
    for (size_t i = 0; i < tile_sizes.size(); i++) {
        if (i + 1 < tile_sizes.size()) {
            symbolic::Expression divisor = symbolic::integer(1);
            for (size_t j = i + 1; j < tile_sizes.size(); j++) {
                divisor = symbolic::mul(divisor, tile_sizes[j]);
            }
            decomp.push_back(symbolic::div(remainder, divisor));
            remainder = symbolic::mod(remainder, divisor);
        } else {
            decomp.push_back(remainder);
        }
    }
    return decomp;
}

LocalStorage::AccessSummary LocalStorage::
    summarize(const StructuredSDFG& sdfg, structured_control_flow::StructuredLoop& loop, const std::string& container) {
    ContainerAccessPolicy policy;
    policy.container = container;
    ContainerAccessVisitor visitor(sdfg, policy);
    visitor.visit(loop.root()); // walks the loop body only
    return AccessSummary{policy.reads, policy.writes, policy.aliased};
}

bool LocalStorage::has_side_effect(structured_control_flow::StructuredLoop& loop) {
    bool found = false;
    for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
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
            if (lib_node->side_effect()) {
                found = true;
                return;
            }
        }
    });
    return found;
}

const analysis::MemoryTileGroup* LocalStorage::tile(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto* groups = analysis_manager.get<analysis::MemoryLayoutAnalysis>().tile_groups(loop, container);
    if (!groups || groups->size() != 1) {
        return nullptr;
    }
    const auto& group = groups->front();
    std::unordered_set<const data_flow::Memlet*> members(group.memlets.begin(), group.memlets.end());

    // Every memlet of the container in the loop body must belong to the group;
    // an unanalyzable (ungrouped) or split memlet makes wholesale rewriting unsafe.
    bool covered = true;
    std::function<void(structured_control_flow::ControlFlowNode&)> walk;
    walk = [&](structured_control_flow::ControlFlowNode& node) {
        if (!covered) {
            return;
        }
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
            auto& dfg = block->dataflow();
            for (auto* access : dfg.data_nodes()) {
                if (access->data() != container) {
                    continue;
                }
                for (auto& memlet : dfg.out_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
                for (auto& memlet : dfg.in_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
            }
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                walk(seq->at(i));
            }
        } else if (auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
            walk(inner->root());
        } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                walk(if_else->at(i).first);
            }
        }
    };
    walk(loop.root());

    return covered ? &group : nullptr;
}

LocalStorage::LocalityPlan LocalStorage::build_locality_plan(
    structured_control_flow::StructuredLoop& loop,
    const TileInfo& tile_info,
    analysis::AnalysisManager& analysis_manager
) {
    LocalityPlan plan;
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    plan.loop_is_outermost = loop_analysis.is_outermost_loop(&loop);

    plan.loop_is_gpu = gpu::is_gpu_schedule(loop.schedule_type());
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(desc);
        if (sl && gpu::is_gpu_schedule(sl->schedule_type())) {
            plan.has_gpu_descendant = true;
            break;
        }
    }

    // A dim is cooperative when its induction variable appears in no tile base:
    // every iteration then addresses the same tile and must stage it together.
    auto is_cooperative = [&](const symbolic::Symbol& indvar) {
        for (const auto& base : tile_info.bases) {
            if (symbolic::uses(base, indvar)) {
                return false;
            }
        }
        return true;
    };

    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        auto* sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(node);
        if (!sloop) {
            continue;
        }
        auto& sched = sloop->schedule_type();
        bool is_gpu = gpu::is_gpu_schedule(sched);
        // Only genuinely parallel loops shape the storage; plain sequential
        // For/Map/Reduce loops don't. A GPU-scheduled Reduce counts here too: its
        // combine is spread across threads exactly like a Map, so it forms a
        // cooperative / per-thread storage level for any read tile inside it.
        if (!is_gpu && sched.category() == structured_control_flow::ScheduleTypeCategory::None) {
            continue;
        }
        LocalityPlan::Dim d;
        d.indvar = sloop->indvar();
        d.is_gpu = is_gpu;
        d.cooperative = is_cooperative(d.indvar);
        d.init = sloop->init();
        if (auto s = sloop->stride(); !s.is_null()) {
            d.stride = s;
        }
        if (is_gpu) {
            const std::string& value = sched.value();
            if (value == "CUDA_Offload" || value == "ROCM_Offload") {
                d.target_level = gpu::gpu_target_level(sched);
                if (gpu::is_grid_level(d.target_level)) {
                    d.level = LocalityPlan::Level::Grid;
                } else if (gpu::is_block_level(d.target_level)) {
                    d.level = LocalityPlan::Level::Block;
                } else {
                    d.level = LocalityPlan::Level::Warp;
                }
                d.parallel_size = gpu::ScheduleType_GPU_Offload::parallel_size(sched);
                d.needs_sync = gpu::ScheduleType_GPU_Offload::nested_sync(sched);
            } else {
                // Legacy CUDA/ROCM: a single fused block-thread dimension.
                d.level = LocalityPlan::Level::Block;
                d.parallel_size = gpu::gpu_block_size(sched);
            }
        }
        plan.dims.push_back(d);
    }

    // Enclosing-scope cooperative staging: the localized loop is itself a GPU map
    // with no enclosing parallel context, and a block-scheduled loop in its body
    // consumes the tile. The tile is staged once per block into shared and reused
    // by every (sibling) consumer below.
    if (plan.dims.empty() && plan.loop_is_gpu && find_block_scheduled_descendant(loop, analysis_manager) != nullptr) {
        plan.enclosing_cooperative = true;
    }

    return plan;
}

bool LocalStorage::is_reduction_accumulator(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto owns = [&](structured_control_flow::ControlFlowNode* node) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (!reduce) {
            return false;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
        return false;
    };
    if (owns(&loop)) {
        return true;
    }
    for (auto* node : loop_analysis.ancestors(&loop)) {
        if (owns(node)) {
            return true;
        }
    }
    for (auto* node : loop_analysis.descendants(&loop)) {
        if (owns(node)) {
            return true;
        }
    }
    return false;
}

bool LocalStorage::collect_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::Reduce*>& out
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto owns = [&](structured_control_flow::ControlFlowNode* node) -> structured_control_flow::Reduce* {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (!reduce) {
            return nullptr;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return reduce;
            }
        }
        return nullptr;
    };

    // A grid-parallel (e.g. split-K Z_GRID) offloaded reduce merges per-block
    // partials across the grid via an atomic writeback rather than in-place.
    auto is_grid_parallel_reduce = [&](structured_control_flow::Reduce* reduce) -> bool {
        const auto& sched = reduce->schedule_type();
        if (!gpu::is_gpu_schedule(sched)) {
            return false;
        }
        const std::string& value = sched.value();
        if (value != "CUDA_Offload" && value != "ROCM_Offload") {
            return false;
        }
        return gpu::is_grid_level(gpu::gpu_target_level(sched));
    };

    // An ancestor Reduce accumulates across iterations *outside* the localized
    // scope. Whether a buffer at loop_ is still legal depends on how its
    // per-outer-iteration writeback composes:
    //   - grid-parallel (GPU) ancestor: per-block partials merge via an atomic
    //     writeback; LocalStorage privatizes the partial and owns the merge (not
    //     retargeted here).
    //   - sequential (non-GPU) ancestor: outer iterations are barrier-separated, so
    //     a read-modify-write copy-in/out around loop_ carries the accumulation
    //     through the global container each iteration (classical BLIS pc loop). A
    //     racy cooperative-CPU writeback is rejected downstream by derive_storage.
    //   - any other (GPU block/warp cooperative) ancestor is combined by the reduce
    //     dispatcher and cannot be localized here.
    for (auto* node : loop_analysis.ancestors(&loop)) {
        if (auto* reduce = owns(node)) {
            if (is_grid_parallel_reduce(reduce)) {
                continue;
            }
            if (!gpu::is_gpu_schedule(reduce->schedule_type())) {
                continue;
            }
            return false;
        }
    }

    // loop_ itself or a descendant Reduce: privatizable only when the reduction is
    // combined sequentially / per-thread. A GPU-offloaded Reduce is combined across
    // threads by the reduce dispatcher, which owns the accumulator staging.
    auto consider = [&](structured_control_flow::Reduce* reduce) -> bool {
        if (gpu::is_gpu_schedule(reduce->schedule_type())) {
            return false;
        }
        out.push_back(reduce);
        return true;
    };
    if (auto* reduce = owns(&loop)) {
        if (!consider(reduce)) {
            return false;
        }
    }
    for (auto* node : loop_analysis.descendants(&loop)) {
        if (auto* reduce = owns(node)) {
            if (!consider(reduce)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<structured_control_flow::Reduce*> LocalStorage::collect_grid_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    std::vector<structured_control_flow::Reduce*> out;
    for (auto* node : loop_analysis.ancestors(&loop)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(node);
        if (reduce == nullptr) {
            continue;
        }
        const auto& sched = reduce->schedule_type();
        if (!gpu::is_gpu_schedule(sched)) {
            continue;
        }
        const std::string& value = sched.value();
        if (value != "CUDA_Offload" && value != "ROCM_Offload") {
            continue;
        }
        if (!gpu::is_grid_level(gpu::gpu_target_level(sched))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                out.push_back(reduce);
                break;
            }
        }
    }
    return out;
}

LocalStorage::Locality LocalStorage::derive_storage(const LocalityPlan& plan, bool container_written) {
    using Level = LocalityPlan::Level;
    // A cooperative CPU-parallel dim means the tile is invariant across the parallel
    // threads (a shared operand, e.g. BLIS ~B). CPU has no shared staging, but a
    // read-only tile is safe by replication: each thread stages its own private copy
    // of the identical tile (redundant packing, no sharing). A written cooperative
    // tile is a genuine cross-thread reduction/race and must reject.
    if (plan.has_cpu_cooperative()) {
        return container_written ? Locality::Reject : Locality::Private;
    }
    // Enclosing-scope staging: a per-block shared row loaded once, reused by the
    // block consumers below. A cooperative write is a reduction (Reduce owns it).
    if (plan.enclosing_cooperative) {
        return container_written ? Locality::Reject : Locality::Shared;
    }
    if (plan.has_gpu_cooperative()) {
        // A cooperative write across threads is a reduction: that is owned by the
        // Reduce node + reduce dispatcher, not LocalStorage.
        if (container_written) {
            bool intra_block_coop = plan.has_cooperative_at(Level::Block) || plan.has_cooperative_at(Level::Warp);
            bool owned_per_thread = false;
            for (const auto& d : plan.gpu_per_thread_dims()) {
                if (d.level == Level::Block || d.level == Level::Warp) {
                    owned_per_thread = true;
                    break;
                }
            }
            // Reject a genuine intra-block/warp reduction (owned by Reduce), or a
            // grid-cooperative write with no per-thread owner (a real cross-block
            // reduction needing atomics/grid sync). But a grid-only "cooperative"
            // write that a finer per-thread block dim already addresses is disjoint
            // per-block output — a private per-thread register tile (fall through).
            if (intra_block_coop || !owned_per_thread) {
                return Locality::Reject;
            }
        } else {
            // A cooperative buffer lives in a device scope inside the kernel, below
            // the outermost loop.
            if (!plan.inside_gpu_kernel() || plan.loop_is_outermost) {
                return Locality::Reject;
            }
            // Storage follows the finest cooperative level that owns a real buffer.
            // A read tile cooperative within a block lives in shared memory even when
            // it is also grid-cooperative: each block redundantly stages its own copy
            // (grid cooperation is replication, not a shared buffer). Only *pure* grid
            // cooperation needs a grid-wide global buffer.
            if (plan.has_cooperative_at(Level::Block)) {
                return Locality::Shared;
            }
            if (plan.has_cooperative_at(Level::Grid)) {
                return Locality::Global;
            }
            // Warp-only cooperation is served by shuffles, not a staged buffer.
            return Locality::Reject;
        }
    }
    // No cooperative dims: a thread-private / sequential buffer. But a host-level
    // loop that is itself GPU-scheduled or wraps a GPU kernel is not a site for
    // a private stack buffer.
    if (!plan.inside_gpu_kernel() && (plan.loop_is_gpu || plan.has_gpu_descendant)) {
        return Locality::Reject;
    }
    return Locality::Private;
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
    if (is_reduction_accumulator(loop_, container_, analysis_manager)) {
        if (!collect_reduction_owners(loop_, container_, analysis_manager, reduce_retargets_)) {
            return false;
        }
        // A grid-parallel ancestor reduce is privatized here (per-block partial) and
        // its cross-block merge becomes an atomic copy-out; record it for apply().
        grid_reduce_owners_ = collect_grid_reduction_owners(loop_, container_, analysis_manager);
        atomic_merge_ = !grid_reduce_owners_.empty();
    }

    // Classify the container's accesses directly from the dataflow.
    auto summary = summarize(sdfg, loop_, container_);
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
        LocalityPlan topo = build_locality_plan(loop_, TileInfo{}, analysis_manager);
        if (topo.enclosing_cooperative) {
            auto consumers = block_scheduled_consumers(loop_, container_, analysis_manager);
            if (consumers.empty()) {
                return false;
            }
            const std::string& sched_value = consumers.front()->schedule_type().value();
            if (sched_value != "CUDA_Offload" && sched_value != "ROCM_Offload") {
                return false;
            }
            const analysis::MemoryTileGroup* ref = tile(*consumers.front(), container_, analysis_manager);
            if (!ref || !is_constant_bounded(ref)) {
                return false;
            }
            auto count = tile_element_count(ref);
            auto budget = symbolic::integer(static_cast<int64_t>(max_tile_elements()));
            if (count.is_null() || !symbolic::is_true(symbolic::Le(count, budget))) {
                return false;
            }
            // Every block consumer must localize the same per-block tile; union their
            // memlets so rewrite_body repoints all of them to the shared buffer.
            group_memlets_.clear();
            for (auto* c : consumers) {
                const analysis::MemoryTileGroup* g = tile(*c, container_, analysis_manager);
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
            storage_type_ = types::StorageType::NV_Shared();
            return true;
        }
    }

    // Resolve the single localizable tile for the whole container.
    auto* group = tile(loop_, container_, analysis_manager);
    if (!group) {
        return false;
    }

    // Extents must be compile-time integer constants.
    if (!is_constant_bounded(group)) {
        return false;
    }

    // Physical capacity: the buffer must fit the target budget.
    auto count = tile_element_count(group);
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
    plan_ = build_locality_plan(loop_, tile_info_, analysis_manager);
    switch (derive_storage(plan_, container_written_)) {
        case Locality::Private:
            storage_type_ = types::StorageType::CPU_Stack();
            break;
        case Locality::Shared: {
            // Cooperative shared-memory path (also handles a per-thread + cooperative
            // mix, e.g. shared-memory GEMM). v1 gate: all enclosing parallel dims are
            // GPU block-level, exactly one is the cooperative (copy) axis, the tile is
            // read-only, and the cooperative Map is the loop's immediate enclosing loop.
            if (container_written_) {
                return false;
            }
            if (plan_.enclosing_cooperative) {
                // Enclosing-scope staging: the localized GPU map's body has a
                // block-scheduled consumer supplying the copy schedule; the buffer is
                // per-block shared, loaded once and reused by every sibling below.
                auto* coop = find_block_scheduled_descendant(loop_, analysis_manager);
                if (!coop) {
                    return false;
                }
                const std::string& sched_value = coop->schedule_type().value();
                if (sched_value != "CUDA_Offload" && sched_value != "ROCM_Offload") {
                    return false;
                }
                storage_type_ = types::StorageType::NV_Shared();
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
            for (const auto& d : plan_.dims) {
                if (!d.is_gpu) {
                    return false; // no CPU dims in the mix
                }
                if (d.level != LocalityPlan::Level::Block && d.level != LocalityPlan::Level::Grid) {
                    return false; // no warp dims
                }
            }
            auto coop_dims = plan_.gpu_cooperative_dims();
            if (coop_dims.empty()) {
                return false;
            }
            if (find_cooperative_offload_map(loop_, coop_dims) == nullptr) {
                return false;
            }
            storage_type_ = types::StorageType::NV_Shared();
            break;
        }
        case Locality::Global:
            // Grid-cooperative tiles need global memory + grid-wide sync, which is
            // not yet implemented.
            return false;
        case Locality::Reject:
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
        for (const auto& d : plan_.gpu_per_thread_dims()) {
            // Only block-level per-thread dims own a buffer slot. A grid per-thread
            // dim selects the block (fixed for all its threads) and is already
            // folded into the tile bases, so it carries no slot.
            if (d.level != LocalityPlan::Level::Block) {
                continue;
            }
            slot_sizes.push_back(d.parallel_size);
            // Within-block thread index = (indvar - init) / stride, then wrapped by
            // the block width. Tiled thread-tile offload maps are NOT normalized
            // (init = per-block base, stride = tile step), so a raw indvar % width
            // aliases distinct threads onto the same slot (e.g. stride 8, width 16
            // over a 128-wide block collapses 16 threads to 2 slots).
            symbolic::Expression tid = symbolic::div(symbolic::sub(d.indvar, d.init), d.stride);
            slot_indices.push_back(symbolic::mod(tid, d.parallel_size));
            slot_indvars.push_back(d.indvar);
            slot_inits.push_back(d.init);
            slot_strides.push_back(d.stride);
        }
    }

    TileBuffer buffer{slot_sizes, tile_info_.varying_sizes()};
    // GPU shared buffers with a per-thread slot prefix use a flat per-slot block so
    // a warp's per-slot accesses hit distinct banks: Padded (stride coprime with 32)
    // by default, or Swizzle (XOR the inner index) when opted in with a constant
    // power-of-two block. CPU/private buffers keep the dense multi-dim layout.
    if (lane_contiguous_ && storage_type_.is_nv_shared()) {
        // Fully flat, thread-linear staging for the CDNA async global->LDS DMA
        // (global_load_lds writes lane-contiguous from a wave-uniform base).
        buffer.layout = TileBuffer::Layout::Linearized;
    } else if (storage_type_.is_nv_shared() && !slot_sizes.empty()) {
        buffer.layout = TileBuffer::Layout::Padded;
        if (swizzle_layout_) {
            auto total = buffer.tile_total_size();
            if (SymEngine::is_a<SymEngine::Integer>(*total)) {
                auto n = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
                if (n > 0 && (n & (n - 1)) == 0) {
                    buffer.layout = TileBuffer::Layout::Swizzle;
                }
            }
        }
    }
    // Cooperative-store conflict avoidance: pad the inner stride to the coop axis's
    // per-warp thread count (mod 32). Compute it from the block dims + the coop
    // copy's axis (A tiles are coop over X, B over Y -> different spans, so a single
    // odd stride leaves one of them 2-way conflicted; this makes both conflict-free).
    if (buffer.layout == TileBuffer::Layout::Padded) {
        if (auto* coop_map = find_cooperative_offload_map(loop_, plan_.gpu_cooperative_dims())) {
            auto block_width = [&](gpu::TargetLevel lvl) -> size_t {
                for (const auto& d : plan_.dims) {
                    if (d.level == LocalityPlan::Level::Block && d.target_level == lvl &&
                        SymEngine::is_a<SymEngine::Integer>(*d.parallel_size)) {
                        return static_cast<size_t>(SymEngine::rcp_static_cast<const SymEngine::Integer>(d.parallel_size)
                                                       ->as_int());
                    }
                }
                return 1;
            };
            const size_t warp = static_cast<size_t>(gpu::gpu_warp_size(coop_map->schedule_type()));
            const size_t bx = block_width(gpu::TargetLevel::X_BLOCK);
            const size_t by = block_width(gpu::TargetLevel::Y_BLOCK);
            const size_t bz = block_width(gpu::TargetLevel::Z_BLOCK);
            // Per-warp thread counts, x fastest in the flat thread index.
            const size_t x_per = std::min(bx, warp);
            const size_t rem_y = std::max<size_t>(1, warp / std::max<size_t>(1, x_per));
            const size_t y_per = std::min(by, rem_y);
            const size_t rem_z = std::max<size_t>(1, rem_y / std::max<size_t>(1, y_per));
            const size_t z_per = std::min(bz, rem_z);
            switch (gpu::gpu_target_level(coop_map->schedule_type())) {
                case gpu::TargetLevel::X_BLOCK:
                    buffer.coop_warp_span = x_per;
                    break;
                case gpu::TargetLevel::Y_BLOCK:
                    buffer.coop_warp_span = y_per;
                    break;
                case gpu::TargetLevel::Z_BLOCK:
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
        if (lane_contiguous_) {
            // Fully flat thread-linear staging: a full-block cooperative copy whose
            // dst is lane-contiguous, so SoftwarePipelining's async global_load_lds
            // lands correctly on CDNA.
            emit_lane_contiguous_copy_in(
                builder,
                analysis_manager,
                *parent,
                buffer,
                buffer_type,
                pointer_type,
                slot_sizes,
                slot_indvars,
                slot_inits,
                slot_strides
            );
        } else if (plan_.enclosing_cooperative) {
            // Stage the row once at the top of the localized GPU map's body; the
            // (sibling) block consumers below read the shared buffer.
            emit_enclosing_cooperative_copy_in(builder, analysis_manager, buffer, buffer_type, pointer_type);
        } else {
            // Read-only cooperative tile: cooperative copy-in + barrier(s), no writeback.
            // A per-thread slot prefix means the shared row is re-staged per kernel
            // coverage iteration, so guard it with a leading barrier too.
            bool leading_barrier = !slot_indices.empty();
            emit_cooperative_copy_in(
                builder, analysis_manager, *parent, buffer, buffer_type, pointer_type, slot_indices, leading_barrier
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
    for (size_t v = 0; v < vdims.size() && v < tile_indices.size(); ++v) {
        size_t d = vdims[v];
        if (d >= tile_info_.maxes.size() || tile_info_.maxes[d].is_null()) {
            continue;
        }
        auto global_d = symbolic::add(tile_info_.bases[d], tile_indices[v]);
        if (!assums.empty() && symbolic::is_le(global_d, tile_info_.maxes[d], params, assums, /*tight=*/true)) {
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
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type,
    bool writeback
) {
    auto varying_dims = tile_info_.varying_dims();
    auto varying_dim_sizes = tile_info_.varying_sizes();

    int index = parent.index(loop_) + (writeback ? 1 : 0);
    auto& scope = writeback ? builder.add_sequence_after(parent, loop_, loop_.debug_info())
                            : builder.add_sequence_before(parent, loop_, loop_.debug_info());
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
    data_flow::Subset buffer_subset = buffer.multi_subset({}, indvars);
    // Element-predicate the global access: the over-approximated tile may address
    // out-of-bounds global memory on ragged blocks. Skip those elements (the
    // buffer slots they'd fill are never consumed — the compute's own boundary
    // handling guards them). Provably in-bounds conjuncts (a fully-covering tile)
    // are dropped so the interior copy vectorizes.
    std::vector<symbolic::Expression> incl_uppers;
    for (const auto& s : varying_dim_sizes) {
        incl_uppers
            .push_back(s.is_null() ? symbolic::Expression(SymEngine::null) : symbolic::sub(s, symbolic::integer(1)));
    }
    auto discharge = build_copy_discharge_assumptions(analysis_manager, parent, indvars, incl_uppers);
    auto guard = boundary_guard(indvars, analysis_manager.get<analysis::AssumptionsAnalysis>().parameters(), discharge);
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

    if (writeback && atomic_merge_) {
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
    } else if (!writeback && atomic_merge_) {
        // Zero-init the per-block partial (it accumulates only this block's k-slice).
        auto& block = builder.add_block(*body);
        auto& zero = builder.add_constant(block, "0", *scalar_type);
        auto& dst = builder.add_access(block, local_name_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {}, *scalar_type);
        builder.add_computational_memlet(block, tasklet, "_out", dst, buffer_subset, buffer_type);
    } else {
        auto& block = builder.add_block(*body);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        if (writeback) {
            auto& src = builder.add_access(block, local_name_);
            auto& dst = builder.add_access(block, container_);
            builder.add_computational_memlet(block, src, tasklet, "_in", buffer_subset, buffer_type);
            builder.add_computational_memlet(block, tasklet, "_out", dst, original_subset, pointer_type);
        } else {
            auto& src = builder.add_access(block, container_);
            auto& dst = builder.add_access(block, local_name_);
            builder.add_computational_memlet(block, src, tasklet, "_in", original_subset, pointer_type);
            builder.add_computational_memlet(block, tasklet, "_out", dst, buffer_subset, buffer_type);
        }
    }

    builder.move_children(scope, parent, index + 1);
    builder.remove_child(parent, index);
}

void LocalStorage::emit_cooperative_copy_in(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& parent,
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type,
    const std::vector<symbolic::Expression>& slot_indices,
    bool leading_barrier
) {
    // Parallelize the copy over a genuine cooperative axis. In a mixed
    // per-thread+cooperative tile the immediate enclosing map is a per-thread axis
    // (the slot axis); striding the copy along it would leave each slot only
    // partially filled. find_cooperative_offload_map picks the cooperative axis
    // whose threads split the tile (guaranteed non-null by can_be_applied).
    auto coop_dims = plan_.gpu_cooperative_dims();
    structured_control_flow::Map* coop_map = find_cooperative_offload_map(loop_, coop_dims);

    // A leading barrier prevents a re-staged (per-thread) tile from being
    // overwritten while the previous coverage iteration's reads are outstanding.
    if (leading_barrier) {
        auto& pre_block = builder.add_block_before(parent, loop_, loop_.debug_info());
        builder.add_library_node<data_flow::BarrierLocalNode>(pre_block, DebugInfo());
    }

    auto c_name = builder.find_new_name("__daisy_ls_coop_" + container_);
    // Int32: sweeps [0, tile_total_size), a constant-bounded tile extent under the
    // max_tile_elements budget; added to 64-bit bases in the global address.
    builder.add_container(c_name, types::Scalar(types::PrimitiveType::Int32));
    auto c = symbolic::symbol(c_name);

    // Each thread fills its own per-thread slot; the cooperative threads split the
    // tile, so the copy Map iterates one slot's worth (the tile) not the whole buffer.
    auto& copy_map = builder.add_map_before(
        parent,
        loop_,
        c,
        symbolic::Lt(c, buffer.tile_total_size()),
        symbolic::integer(0),
        symbolic::add(c, symbolic::integer(1)),
        coop_map->schedule_type(),
        loop_.debug_info()
    );

    auto decomp = buffer.delinearize_tile(c);

    // Element-predicate the cooperative global read so ragged blocks never read
    // out-of-bounds; skipped slots are never consumed (guarded by the compute).
    // Provably in-bounds conjuncts (a fully-covering tile) are dropped so the hot
    // cooperative load vectorizes instead of running under a predicate.
    data_flow::Subset coop_original = tile_info_.original_subset(decomp);
    std::vector<symbolic::Expression> coop_uppers;
    {
        auto total = buffer.tile_total_size();
        coop_uppers.push_back(
            total.is_null() ? symbolic::Expression(SymEngine::null) : symbolic::sub(total, symbolic::integer(1))
        );
    }
    auto coop_discharge = build_copy_discharge_assumptions(analysis_manager, parent, {c}, coop_uppers);
    auto coop_guard =
        boundary_guard(decomp, analysis_manager.get<analysis::AssumptionsAnalysis>().parameters(), coop_discharge);
    structured_control_flow::Sequence* coop_body = &copy_map.root();
    if (!symbolic::is_true(coop_guard)) {
        auto& if_else = builder.add_if_else(copy_map.root(), loop_.debug_info());
        coop_body = &builder.add_case(if_else, coop_guard, loop_.debug_info());
    }

    auto& block = builder.add_block(*coop_body);
    auto& src = builder.add_access(block, container_);
    auto& dst = builder.add_access(block, local_name_);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Per-thread slot row followed by the tile offset. A flat padded inner block
    // takes the coop flat index directly (keeps the copy vectorizable); MultiDim,
    // Swizzle, and Linearized route through multi_subset.
    data_flow::Subset dst_subset;
    if (buffer.layout == TileBuffer::Layout::Padded) {
        dst_subset = slot_indices;
        dst_subset.push_back(c);
    } else {
        dst_subset = buffer.multi_subset(slot_indices, decomp);
    }
    builder.add_computational_memlet(block, src, tasklet, "_in", coop_original, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", dst, dst_subset, buffer_type);

    // Barrier so every thread's load is visible before the tile is consumed.
    auto& barrier_block = builder.add_block_before(parent, loop_, loop_.debug_info());
    builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block, DebugInfo());
}

void LocalStorage::emit_enclosing_cooperative_copy_in(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type
) {
    // A block-scheduled consumer in the body supplies the copy schedule (verified in
    // can_be_applied). The staged row is loaded once at the top of the body, then a
    // barrier makes it visible before the sibling consumers read it.
    auto* coop = find_block_scheduled_descendant(loop_, analysis_manager);
    auto& body = loop_.root();
    auto& first = body.at(0);

    auto c_name = builder.find_new_name("__daisy_ls_coop_" + container_);
    // Int32: sweeps [0, tile_total_size), a constant-bounded tile extent under the
    // max_tile_elements budget; added to 64-bit bases in the global address.
    builder.add_container(c_name, types::Scalar(types::PrimitiveType::Int32));
    auto c = symbolic::symbol(c_name);

    // Copy map: the block cooperatively splits the tile (one shared row, no slots).
    auto& copy_map = builder.add_map_before(
        body,
        first,
        c,
        symbolic::Lt(c, buffer.tile_total_size()),
        symbolic::integer(0),
        symbolic::add(c, symbolic::integer(1)),
        coop->schedule_type(),
        loop_.debug_info()
    );

    auto decomp = buffer.delinearize_tile(c);
    auto& block = builder.add_block(copy_map.root());
    auto& src = builder.add_access(block, container_);
    auto& dst = builder.add_access(block, local_name_);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    data_flow::Subset dst_subset = buffer.multi_subset({}, decomp);
    builder.add_computational_memlet(block, src, tasklet, "_in", tile_info_.original_subset(decomp), pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", dst, dst_subset, buffer_type);

    // Trailing barrier (after the copy, before the consumers) — no leading barrier,
    // the shared row is loaded once per block at body entry.
    auto& barrier_block = builder.add_block_before(body, first, loop_.debug_info());
    builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block, DebugInfo());
}

void LocalStorage::emit_lane_contiguous_copy_in(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& parent,
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type,
    const std::vector<symbolic::Expression>& slot_sizes,
    const std::vector<symbolic::Expression>& slot_indvars,
    const std::vector<symbolic::Expression>& slot_inits,
    const std::vector<symbolic::Expression>& slot_strides
) {
    // Leading barrier: the flat tile is re-staged each panel, so guard the overwrite
    // against the previous panel's outstanding reads.
    auto& pre = builder.add_block_before(parent, loop_, loop_.debug_info());
    builder.add_library_node<data_flow::BarrierLocalNode>(pre, DebugInfo());

    // Flat thread id over the full block (x fastest) + block size, from every
    // block-level offload dim (both cooperative and per-thread/slot). The whole
    // block cooperatively sweeps the tile in thread-linear order.
    struct BlockDim {
        symbolic::Expression tid;
        long long width;
        int order;
    };
    std::vector<BlockDim> bdims;
    for (const auto& d : plan_.dims) {
        if (d.level != LocalityPlan::Level::Block) {
            continue;
        }
        long long w =
            static_cast<long long>(SymEngine::rcp_static_cast<const SymEngine::Integer>(d.parallel_size)->as_int());
        auto tid = symbolic::mod(symbolic::div(symbolic::sub(d.indvar, d.init), d.stride), d.parallel_size);
        bdims.push_back({tid, w, static_cast<int>(d.target_level)});
    }
    std::sort(bdims.begin(), bdims.end(), [](const BlockDim& a, const BlockDim& b) { return a.order < b.order; });
    symbolic::Expression flat_tid = symbolic::integer(0);
    long long blk = 1;
    for (const auto& b : bdims) {
        flat_tid = symbolic::add(flat_tid, symbolic::mul(b.tid, symbolic::integer(blk)));
        blk *= b.width;
    }

    long long total =
        static_cast<long long>(SymEngine::rcp_static_cast<const SymEngine::Integer>(buffer.total_size())->as_int());

    // Coverage loop: each thread strides the flat tile from flat_tid by blk. The
    // loop index c is the flat position itself (init = flat_tid, step = blk), so it
    // stays an opaque symbol in the delinearized indices below. This is essential:
    // the gather substitutes the slot indvar (threadIdx) with its delinearized value,
    // and if c were the expanded (flat_tid + blk*it) form it would still contain the
    // slot indvar inside the delinearized tile indices, which the substitution would
    // then corrupt. The `c < total` bound also subsumes the ragged-tail guard.
    auto c_name = builder.find_new_name("__daisy_ls_lc_" + container_);
    builder.add_container(c_name, types::Scalar(types::PrimitiveType::Int32));
    auto c = symbolic::symbol(c_name);
    auto& cov = builder.add_for_before(
        parent,
        loop_,
        c,
        symbolic::Lt(c, symbolic::integer(total)),
        flat_tid,
        symbolic::add(c, symbolic::integer(blk)),
        loop_.debug_info()
    );
    structured_control_flow::Sequence* copy_body = &cov.root();

    // Delinearize c over [slot_sizes ++ tile_sizes] (row-major): slot_idx select the
    // per-thread base offset, tile_idx the within-tile position.
    std::vector<symbolic::Expression> combined = slot_sizes;
    combined.insert(combined.end(), buffer.tile_sizes.begin(), buffer.tile_sizes.end());
    std::vector<symbolic::Expression> idx(combined.size());
    symbolic::Expression rem = c;
    for (int i = static_cast<int>(combined.size()) - 1; i >= 0; i--) {
        idx[i] = symbolic::mod(rem, combined[i]);
        rem = symbolic::div(rem, combined[i]);
    }
    std::vector<symbolic::Expression> slot_values;
    for (size_t s = 0; s < slot_sizes.size(); s++) {
        slot_values.push_back(symbolic::add(slot_inits.at(s), symbolic::mul(slot_strides.at(s), idx.at(s))));
    }
    std::vector<symbolic::Expression> tile_idx(idx.begin() + slot_sizes.size(), idx.end());
    auto src_subset = tile_info_.flat_original_subset(slot_indvars, slot_values, tile_idx);

    auto& block = builder.add_block(*copy_body);
    auto& src = builder.add_access(block, container_);
    auto& dst = builder.add_access(block, local_name_);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, src, tasklet, "_in", src_subset, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", dst, {c}, buffer_type);

    // Trailing barrier: publish the staged tile before the consumers read it.
    auto& post = builder.add_block_before(parent, loop_, loop_.debug_info());
    builder.add_library_node<data_flow::BarrierLocalNode>(post, DebugInfo());
}

void LocalStorage::rewrite_body(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const std::vector<symbolic::Expression>& slot_indices
) {
    // v1 guarantees single-group full coverage, so every group memlet is rewritten
    // and its access node renamed (no split-node handling needed).
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    for_each_block(loop_.root(), [&](structured_control_flow::Block& block) {
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
                memlet.set_subset(buffer.multi_subset(slot_indices, tile_info_.local_index(acc->subset)));
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

    return LocalStorage(*loop, *access_node, swizzle_layout, lane_contiguous);
}

} // namespace transformations
} // namespace sdfg
