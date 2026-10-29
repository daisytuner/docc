#include "sdfg/transformations/stream_k.h"

#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_tile_arrival_node.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

#include <functional>
#include <map>
#include <optional>

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

namespace {

using namespace structured_control_flow;

bool is_grid_level(gpu::TargetLevel level) {
    return level == gpu::TargetLevel::X_GRID || level == gpu::TargetLevel::Y_GRID || level == gpu::TargetLevel::Z_GRID;
}

/// Returns true if the given loop is a grid-level GPU offload (Map) node. Safe to ask on non-offloaded loops
bool is_grid_offloaded(const StructuredLoop& loop) {
    try {
        auto tgt_level = gpu::ScheduleType_GPU_Offload::target_level_if_exists(loop.schedule_type());
        return tgt_level.has_value() && is_grid_level(tgt_level.value());
    } catch (...) {
        return false; // schedule is not a GPU offload / target_level unset
    }
}

// First Reduce node in a subtree (pre-order). Recurses through sequences, other
// loops (Map/For) and both cases of an if/else, but treats a Reduce as a leaf --
// it is the reduction (fold) axis we are looking for.
Reduce* find_reduce(ControlFlowNode& node) {
    if (auto* reduce = dynamic_cast<Reduce*>(&node)) {
        return reduce;
    }
    if (auto* seq = dynamic_cast<Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); ++i) {
            if (auto* found = find_reduce(seq->at(i))) {
                return found;
            }
        }
    } else if (auto* loop = dynamic_cast<StructuredLoop*>(&node)) {
        return find_reduce(loop->root());
    } else if (auto* if_else = dynamic_cast<IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i) {
            if (auto* found = find_reduce(if_else->at(i).first)) {
                return found;
            }
        }
    }
    return nullptr;
}

// A compile-time-constant, strictly-positive trip count.
bool is_constant_trip(const symbolic::Expression& iterations) {
    if (!SymEngine::is_a<SymEngine::Integer>(*iterations)) {
        return false;
    }
    return !SymEngine::rcp_static_cast<const SymEngine::Integer>(iterations)->is_negative() &&
           !SymEngine::eq(*iterations, *symbolic::zero());
}


std::optional<int64_t> constant_trip(const symbolic::Expression& iterations) {
    if (!is_constant_trip(iterations)) {
        return std::nullopt;
    }
    return SymEngine::rcp_static_cast<const SymEngine::Integer>(iterations)->as_int();
}

struct GridChain {
    std::vector<Map*> maps;
    Sequence* block_parent = nullptr;
    ControlFlowNode* block_subtree = nullptr;
};

/// The perfectly nested grid-level output-tile Maps rooted at @p anchor and the first non-grid node below them.
std::optional<GridChain> grid_chain(StructuredLoop& anchor) {
    GridChain chain;
    StructuredLoop* cur = &anchor;
    while (true) {
        auto* m = dynamic_cast<Map*>(cur);
        if (m == nullptr || m->root().size() != 1) {
            return std::nullopt;
        }
        chain.maps.push_back(m);
        ControlFlowNode& child = m->root().at(0);
        if (auto* child_map = dynamic_cast<Map*>(&child); child_map != nullptr && is_grid_offloaded(*child_map)) {
            cur = child_map;
            continue;
        }
        chain.block_parent = &m->root();
        chain.block_subtree = &child;
        return chain;
    }
}

/// MMA accumulator fragments updated in @p root, keyed by container, with the MMA node updating them.
std::map<std::string, const gpu::GpuMmaMatmulNode*> mma_accumulators(ControlFlowNode& root) {
    std::map<std::string, const gpu::GpuMmaMatmulNode*> accs;
    std::function<void(ControlFlowNode&)> visit = [&](ControlFlowNode& node) {
        if (auto* block = dynamic_cast<Block*>(&node)) {
            auto& graph = block->dataflow();
            for (auto* lib : graph.library_nodes()) {
                auto* mma = dynamic_cast<const gpu::GpuMmaMatmulNode*>(lib);
                if (mma == nullptr) {
                    continue;
                }
                for (auto& edge : graph.in_edges(*lib)) {
                    auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
                    if (access && edge.dst_conn() == lib->input(gpu::GpuMmaMatmulNode::Y_INPUT_IDX)) {
                        accs[access->data()] = mma;
                    }
                }
            }
        } else if (auto* seq = dynamic_cast<Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); ++i) {
                visit(seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<StructuredLoop*>(&node)) {
            visit(loop->root());
        } else if (auto* if_else = dynamic_cast<IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); ++i) {
                visit(if_else->at(i).first);
            }
        }
    };
    visit(root);
    return accs;
}

/// The MMA fragment accumulators of @p reduce: nullopt if its body updates none, empty if some reduction
/// container is not one (mixed reductions are unsupported).
std::optional<std::map<std::string, const gpu::GpuMmaMatmulNode*>> fragment_accumulators(Reduce& reduce) {
    auto accs = mma_accumulators(reduce.root());
    if (accs.empty()) {
        return std::nullopt;
    }
    std::map<std::string, const gpu::GpuMmaMatmulNode*> result;
    for (const auto& info : reduce.reductions()) {
        auto it = accs.find(info.container);
        if (it == accs.end()) {
            return std::map<std::string, const gpu::GpuMmaMatmulNode*>{};
        }
        result.insert(*it);
    }
    return result;
}

/// The block-level Maps enclosing @p reduce below @p block_subtree, outermost first; nullopt if anything but
/// constant-trip Maps and Sequences lies in between.
std::optional<std::vector<Map*>> wave_maps(Reduce& reduce, ControlFlowNode& block_subtree) {
    std::vector<Map*> maps;
    for (auto* node : ControlFlowNode::parent_chain(reduce)) {
        if (dynamic_cast<Sequence*>(node)) {
            continue;
        }
        auto* map = dynamic_cast<Map*>(node);
        if (map == nullptr || !is_constant_trip(map->num_iterations())) {
            return std::nullopt;
        }
        maps.insert(maps.begin(), map);
        if (node == &block_subtree) {
            return maps;
        }
    }
    return std::nullopt;
}

/// Number of blocks splitting each tile's panel range evenly, if @p num_blocks is such a decomposition.
std::optional<int64_t> aligned_splits(int64_t tiles, int64_t panels, int64_t num_blocks) {
    if (num_blocks <= 0 || num_blocks % tiles != 0) {
        return std::nullopt;
    }
    int64_t splits = num_blocks / tiles;
    if (panels % splits != 0) {
        return std::nullopt;
    }
    return splits;
}
} // namespace

StreamK::StreamK(structured_control_flow::StructuredLoop& grid_loop, size_t num_blocks)
    : grid_loop_(grid_loop), num_blocks_(num_blocks) {
}

std::string StreamK::name() const {
    return "StreamK";
}

bool StreamK::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    (void) builder;
    (void) analysis_manager;

    // (1) Anchor is a GPU-offloaded parallel output-tile band (a grid-level Map).
    auto* grid_map = dynamic_cast<structured_control_flow::Map*>(&grid_loop_);
    if (grid_map == nullptr || !is_grid_offloaded(grid_loop_)) {
        return false;
    }

    // (4a) The tile band's trip must be a compile-time constant (needed to form
    // the flattened iteration count and the decode).
    if (!is_constant_trip(grid_loop_.num_iterations())) {
        return false;
    }

    // (2) The band must contain a reduction axis to fold -- discovered via the
    // first-class Reduce node, not by pattern-matching the microkernel.
    auto* reduce = find_reduce(grid_loop_.root());
    if (reduce == nullptr || reduce->reductions().empty()) {
        return false;
    }

    // (3) Every reduction must be associative with a hardware atomic (Add). This
    // is what makes splitting the reduction axis across blocks legal.
    for (const auto& info : reduce->reductions()) {
        if (info.operation != structured_control_flow::ReductionOperation::Add) {
            return false;
        }
    }

    // (4b) The reduction extent (k-panel count) must be a compile-time constant
    // and uniform, so the flat (tile x panel) decode is well-defined.
    if (!is_constant_trip(reduce->num_iterations())) {
        return false;
    }

    auto fragments = fragment_accumulators(*reduce);
    if (fragments.has_value()) {
        if (fragments->empty()) {
            return false;
        }
        auto chain = grid_chain(grid_loop_);
        if (!chain.has_value() || !wave_maps(*reduce, *chain->block_subtree).has_value()) {
            return false;
        }
        int64_t tiles = 1;
        for (auto* map : chain->maps) {
            auto trip = constant_trip(map->num_iterations());
            if (!trip.has_value()) {
                return false;
            }
            tiles *= *trip;
        }
        auto panels = constant_trip(reduce->num_iterations());
        if (!aligned_splits(tiles, *panels, static_cast<int64_t>(num_blocks_)).has_value()) {
            return false;
        }
    }

    return true;
}

void StreamK::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    namespace sym = symbolic;
    using namespace structured_control_flow;

    if (auto* reduce = find_reduce(grid_loop_.root())) {
        if (auto fragments = fragment_accumulators(*reduce); fragments.has_value() && !fragments->empty()) {
            apply_fragment_split(builder, analysis_manager, *reduce);
            return;
        }
    }

    // (0) Walk the grid-level output-tile Maps from the anchor down to the first
    // non-grid child. That child -- the block-map + K-reduction subtree (or, in
    // the degenerate case, the reduction itself) -- becomes the worker body.
    std::vector<Map*> grid_chain;
    Sequence* block_parent = nullptr;
    ControlFlowNode* block_subtree = nullptr;
    {
        StructuredLoop* cur = &grid_loop_;
        while (true) {
            auto* m = dynamic_cast<Map*>(cur);
            if (m == nullptr) {
                throw std::runtime_error("StreamK::apply: grid chain node is not a Map");
            }
            grid_chain.push_back(m);
            Sequence& body = m->root();
            if (body.size() != 1) {
                throw std::runtime_error("StreamK::apply: grid map is not perfectly nested (expected one child)");
            }
            ControlFlowNode& child = body.at(0);
            if (auto* child_map = dynamic_cast<Map*>(&child); child_map != nullptr && is_grid_offloaded(*child_map)) {
                cur = child_map; // another grid-level output-tile map
                continue;
            }
            block_parent = &body;
            block_subtree = &child;
            break;
        }
    }
    const size_t num_dims = grid_chain.size();

    // (1) The reduction to fold lives below the block maps (the KC-panel loop).
    Reduce* reduce = find_reduce(*block_subtree);
    if (reduce == nullptr) {
        throw std::runtime_error("StreamK::apply: no reduction found below the tile band");
    }
    const auto reductions = reduce->reductions();

    // (2) Capture per-dim old indvar, step (= update - indvar, since the tile
    // loops carry offsets not normalized indices), and trip -- before mutating.
    std::vector<sym::Symbol> old_indvars;
    std::vector<sym::Expression> steps;
    std::vector<sym::Expression> trips;
    for (auto* g : grid_chain) {
        old_indvars.push_back(g->indvar());
        steps.push_back(sym::sub(g->update(), g->indvar()));
        trips.push_back(g->num_iterations());
    }
    sym::Expression panels = reduce->num_iterations();
    sym::Expression panel_step = sym::sub(reduce->update(), reduce->indvar());

    sym::Expression tiles = sym::integer(1);
    for (auto& t : trips) {
        tiles = sym::mul(tiles, t);
    }
    sym::Expression total = sym::mul(tiles, panels);
    sym::Integer nblocks = sym::integer(static_cast<int>(num_blocks_));
    sym::Expression one = sym::integer(1);

    auto bid = sym::symbol("__streamk_bid");
    auto t = sym::symbol("__streamk_tile");
    auto merge = sym::symbol("__streamk_merge");
    // total and nblocks are compile-time constants here, so we can bound the
    // worker decode arithmetic and narrow to a signed 32-bit int when it provably
    // fits (fewer live registers -> higher occupancy). A 1<<30 ceiling leaves
    // headroom for the intermediate products below.
    auto fits_int32 = [](const sym::Expression& e) {
        return SymEngine::is_a<SymEngine::Integer>(*e) &&
               symbolic::is_true(symbolic::Le(e, symbolic::integer(1 << 30)));
    };
    // bid and merge only reach the K-panel range (<= K) or a 1-trip loop; their
    // largest intermediate is (bid+1)*total <= nblocks*total, so gate on that.
    // __streamk_tile stays Int64: it decodes the OUTPUT tile, which enters global
    // addresses scaled by a leading dimension not visible here.
    bool narrow_bid = fits_int32(sym::mul(total, nblocks));
    // Signed: the segment decode subtracts (iter_begin - t*panels), negative for
    // every tile past the block's first -- an unsigned type would underflow to a
    // huge value and skip the segment (only the first tile of a block would run).
    types::Scalar tile_type(types::PrimitiveType::Int64);
    types::Scalar bid_type(narrow_bid ? types::PrimitiveType::Int32 : types::PrimitiveType::Int64);
    builder.add_container("__streamk_bid", bid_type);
    builder.add_container("__streamk_tile", tile_type);
    builder.add_container("__streamk_merge", bid_type);

    // This block's equal contiguous slice of the flat (tile x panel) space. bid
    // is emitted with its signed container type (see idx_type), so the segment
    // subtraction (iter_begin - t*panels) is signed and does not underflow for
    // tiles past the block's first; keeping it inline (not a materialized symbol)
    // lets bound analysis bound bid in [0, nblocks) from the grid map and thereby
    // discharge the cooperative-copy boundary guards.
    sym::Expression iter_begin = sym::div(sym::mul(bid, total), nblocks);
    sym::Expression iter_end = sym::div(sym::mul(sym::add(bid, one), total), nblocks);
    sym::Expression zero = sym::integer(0);

    // The worker iterates the TILES this block touches with an AFFINE update
    // (t += 1). An affine stride lets bound analysis prove t in [t_begin, ...),
    // hence bound the imod/idiv tile decode -- which is what lets the cooperative-
    // copy boundary guards discharge (a non-affine `iter` jump does not).
    sym::Expression t_begin = sym::div(iter_begin, panels);
    sym::Expression t_base = sym::mul(t, panels);
    // Panel sub-range of tile t inside the block's flat slice, in the reduction's
    // offset domain: [max(0, iter_begin - t*panels), min(panels, iter_end - t*panels)).
    sym::Expression klo = sym::max(zero, sym::sub(iter_begin, t_base));
    sym::Expression khi = sym::min(panels, sym::sub(iter_end, t_base));

    // Per-dim tile index from t: idx_i = (t / prod(trips[i+1..])) % trips[i],
    // mapped back to the loop's offset domain via * step_i.
    std::vector<sym::Expression> tile_offset(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
        sym::Expression divisor = sym::integer(1);
        for (size_t j = i + 1; j < num_dims; ++j) {
            divisor = sym::mul(divisor, trips[j]);
        }
        sym::Expression index = sym::mod(sym::div(t, divisor), trips[i]);
        tile_offset[i] = sym::mul(index, steps[i]);
    }

    // (3) The atomic-merge point: a degenerate (single-iteration) grid-level
    // reduction over the accumulator, above the block maps. LocalStorage detects
    // it as the grid-parallel owner of C and turns the register-tile writeback
    // into one atomicAdd per worker segment. Built from the grid map's own offload
    // schedule, retargeted to Z_GRID / parallel_size 1 / Global.
    auto merge_sched = grid_chain[0]->schedule_type();
    gpu::ScheduleType_GPU_Offload::target_level(merge_sched, gpu::TargetLevel::Z_GRID);
    gpu::ScheduleType_GPU_Offload::parallel_size(merge_sched, sym::integer(1));
    gpu::ScheduleType_GPU_Offload::partial_storage(merge_sched, gpu::ReduceStrategy::Global);

    // (4) Repurpose the outermost grid map as the fixed persistent grid over bid.
    Map* outer = grid_chain[0];
    Sequence& outer_body = outer->root();
    builder.update_loop(*outer, bid, sym::Lt(bid, nblocks), sym::integer(0), sym::add(bid, one));
    auto grid_sched = outer->schedule_type();
    gpu::ScheduleType_GPU_Offload::parallel_size(grid_sched, nblocks);
    gpu::ScheduleType_GPU_Offload::target_level(grid_sched, gpu::TargetLevel::X_GRID);
    builder.update_schedule_type(*outer, grid_sched);

    // (5) Insert the affine worker tile loop, the degenerate merge reduce inside
    // it, then move the block-map + reduction subtree under the merge; drop the
    // now redundant inner grid maps (their indexing lives in the decode).
    auto& worker = builder.add_for(outer_body, t, sym::Lt(t_base, iter_end), t_begin, sym::add(t, one));
    // Barrier at the top of every worker iteration: consecutive tile segments
    // reuse the same cooperative shared buffers, so the previous segment's reads
    // must complete before this one restages (WAR hazard across the worker walk).
    auto& sync_block = builder.add_block(worker.root());
    builder.add_library_node<data_flow::BarrierLocalNode>(sync_block, DebugInfo());
    auto& kmerge = builder.add_reduce(
        worker.root(), merge, sym::Lt(merge, one), sym::integer(0), sym::add(merge, one), reductions, merge_sched
    );
    builder.move_child(*block_parent, block_parent->index(*block_subtree), kmerge.root());
    if (num_dims > 1) {
        builder.remove_child(outer_body, outer_body.index(*grid_chain[1]));
    }

    // (6) Re-bound the KC-panel reduction to tile t's panel sub-range [klo, khi)
    // (in the reduction's offset domain). The register accumulator now spans the
    // whole contiguous segment, so exactly one atomic merge (at kmerge) fires per
    // segment -- coarse atomics instead of one per fixed grid-Z chunk.
    sym::Expression klo_off = sym::mul(klo, panel_step);
    sym::Expression khi_off = sym::mul(khi, panel_step);
    builder.update_loop(*reduce, reduce->indvar(), sym::Lt(reduce->indvar(), khi_off), klo_off, reduce->update());

    // (7) Rebind each old grid offset to its decoded tile offset.
    for (size_t i = 0; i < num_dims; ++i) {
        builder.replace_symbols(old_indvars[i], tile_offset[i]);
    }

    analysis_manager.invalidate_all();
}

void StreamK::apply_fragment_split(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& reduce
) {
    namespace sym = symbolic;
    using namespace structured_control_flow;

    auto chain = grid_chain(grid_loop_).value();
    auto waves = wave_maps(reduce, *chain.block_subtree).value();
    auto accumulators = fragment_accumulators(reduce).value();
    const auto reductions = reduce.reductions();
    const size_t num_dims = chain.maps.size();

    std::vector<sym::Symbol> old_indvars;
    std::vector<sym::Expression> steps;
    std::vector<int64_t> trips;
    int64_t tiles = 1;
    for (auto* g : chain.maps) {
        old_indvars.push_back(g->indvar());
        steps.push_back(sym::sub(g->update(), g->indvar()));
        trips.push_back(constant_trip(g->num_iterations()).value());
        tiles *= trips.back();
    }
    const int64_t panels = constant_trip(reduce.num_iterations()).value();
    const int64_t num_blocks = static_cast<int64_t>(num_blocks_);
    const int64_t splits = aligned_splits(tiles, panels, num_blocks).value();
    const int64_t panels_per_block = panels / splits;
    sym::Expression panel_step = sym::sub(reduce.update(), reduce.indvar());
    sym::Expression panel_init = reduce.init();

    sym::Expression wave = sym::zero();
    int64_t num_waves = 1;
    for (auto* map : waves) {
        auto trip = constant_trip(map->num_iterations()).value();
        auto index = sym::div(sym::sub(map->indvar(), map->init()), sym::sub(map->update(), map->indvar()));
        wave = sym::add(sym::mul(wave, sym::integer(trip)), index);
        num_waves *= trip;
    }

    auto bid = sym::symbol(builder.find_new_name("__streamk_bid"));
    const bool narrow = num_blocks <= (int64_t(1) << 30);
    builder
        .add_container(bid->get_name(), types::Scalar(narrow ? types::PrimitiveType::Int32 : types::PrimitiveType::Int64));
    sym::Expression tile = sym::div(bid, sym::integer(splits));
    sym::Expression split = sym::mod(bid, sym::integer(splits));

    Map* outer = chain.maps[0];
    builder.update_loop(*outer, bid, sym::Lt(bid, sym::integer(num_blocks)), sym::zero(), sym::add(bid, sym::one()));
    auto grid_sched = outer->schedule_type();
    gpu::ScheduleType_GPU_Offload::parallel_size(grid_sched, sym::integer(num_blocks));
    gpu::ScheduleType_GPU_Offload::target_level(grid_sched, gpu::TargetLevel::X_GRID);
    builder.update_schedule_type(*outer, grid_sched);
    if (num_dims > 1) {
        Sequence& outer_body = outer->root();
        builder.move_child(*chain.block_parent, chain.block_parent->index(*chain.block_subtree), outer_body);
        builder.remove_child(outer_body, outer_body.index(*chain.maps[1]));
    }

    sym::Expression k_begin =
        sym::add(panel_init, sym::mul(sym::mul(split, sym::integer(panels_per_block)), panel_step));
    sym::Expression k_end = sym::add(k_begin, sym::mul(sym::integer(panels_per_block), panel_step));
    builder.update_loop(reduce, reduce.indvar(), sym::Lt(reduce.indvar(), k_end), k_begin, reduce.update());

    for (size_t i = 0; i < num_dims; ++i) {
        int64_t divisor = 1;
        for (size_t j = i + 1; j < num_dims; ++j) {
            divisor *= trips[j];
        }
        sym::Expression index = sym::mod(sym::div(tile, sym::integer(divisor)), sym::integer(trips[i]));
        builder.replace_symbols(old_indvars[i], sym::mul(index, steps[i]));
    }

    if (splits > 1) {
        build_fragment_merge(builder, reduce, reductions, accumulators, bid, tile, wave, num_waves, splits, tiles);
    }

    analysis_manager.invalidate_all();
}

void StreamK::build_fragment_merge(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Reduce& reduce,
    const std::vector<structured_control_flow::ReductionInfo>& reductions,
    const std::map<std::string, const gpu::GpuMmaMatmulNode*>& accumulators,
    const symbolic::Symbol& bid,
    const symbolic::Expression& tile,
    const symbolic::Expression& wave,
    int64_t num_waves,
    int64_t splits,
    int64_t tiles
) {
    namespace sym = symbolic;
    using namespace structured_control_flow;

    auto& sdfg = builder.subject();
    auto& parent = static_cast<Sequence&>(*reduce.get_parent());
    const auto debug_info = reduce.debug_info();
    const auto* first = accumulators.at(reductions.front().container);
    const auto block_size = first->mma_block_size();
    const auto acc_type = first->acc_type();
    const auto impl_type = first->implementation_type();
    const int64_t num_fragments = static_cast<int64_t>(reductions.size());
    const int64_t fragment_elems = static_cast<int64_t>(block_size.m) * block_size.n;

    types::Scalar acc_scalar(acc_type);
    types::Pointer acc_pointer(acc_scalar);
    types::Scalar i32(types::PrimitiveType::Int32);
    auto ws_name = builder.find_new_name("__streamk_workspace");
    types::Array ws_type(
        types::StorageType::GPU_Static(),
        16,
        "",
        acc_scalar,
        sym::integer(static_cast<int64_t>(num_blocks_) * num_waves * num_fragments * fragment_elems)
    );
    builder.add_container(ws_name, ws_type);
    auto cnt_name = builder.find_new_name("__streamk_arrivals");
    types::Array cnt_type(types::StorageType::GPU_Static(), 4, "", i32, sym::integer(tiles));
    builder.add_container(cnt_name, cnt_type);
    auto last = sym::symbol(builder.find_new_name("__streamk_last"));
    builder.add_container(last->get_name(), i32);
    auto peer = sym::symbol(builder.find_new_name("__streamk_split"));
    builder.add_container(peer->get_name(), i32);

    auto slot = [&](const sym::Expression& block, int64_t fragment) {
        auto linear = sym::
            add(sym::mul(sym::add(sym::mul(block, sym::integer(num_waves)), wave), sym::integer(num_fragments)),
                sym::integer(fragment));
        return gpu::GpuMmaFromMemoryLayout{
            .offset = sym::mul(linear, sym::integer(fragment_elems)),
            .ldstride = sym::integer(block_size.n),
            .layout = gpu::MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR
        };
    };
    auto connect = [&](Block& block,
                       const std::string& container,
                       data_flow::LibraryNode& node,
                       int input,
                       const types::IType& type) {
        auto& access = builder.add_access(block, container, debug_info);
        builder.add_computational_memlet(block, access, node, node.input(input), {}, type, debug_info);
    };

    auto& store_block = builder.add_block_after(parent, reduce, debug_info);
    for (int64_t f = 0; f < num_fragments; ++f) {
        auto& store = builder.add_library_node<gpu::GpuMmaFragmentStoreNode>(
            store_block, debug_info, block_size, gpu::MmaFragmentType::C, slot(bid, f), acc_type, impl_type
        );
        connect(store_block, reductions[f].container, store, gpu::GpuMmaFragmentStoreNode::FRAG_INPUT_IDX, acc_pointer);
        connect(store_block, ws_name, store, gpu::GpuMmaFragmentStoreNode::PTR_INPUT_IDX, ws_type);
    }

    auto& arrival_block = builder.add_block_after(parent, store_block, debug_info);
    auto& arrival =
        builder.add_library_node<gpu::GpuTileArrivalNode>(arrival_block, debug_info, tile, sym::integer(splits));
    connect(arrival_block, cnt_name, arrival, gpu::GpuTileArrivalNode::COUNTER_INPUT_IDX, cnt_type);
    auto& last_access = builder.add_access(arrival_block, last->get_name(), debug_info);
    builder.add_computational_memlet(arrival_block, arrival, "_last", last_access, {}, i32, debug_info);

    auto& election = builder.add_if_else_after(parent, arrival_block, debug_info);
    auto& winner = builder.add_case(election, sym::Ne(last, sym::zero()), debug_info);

    auto& zero_block = builder.add_block(winner, debug_info);
    for (int64_t f = 0; f < num_fragments; ++f) {
        auto& fill =
            builder.add_library_node<gpu::GpuMmaFillNode>(zero_block, debug_info, block_size, acc_type, impl_type);
        connect(zero_block, reductions[f].container, fill, gpu::GpuMmaFillNode::Y_INPUT_IDX, acc_pointer);
    }

    std::vector<std::string> partials;
    for (int64_t f = 0; f < num_fragments; ++f) {
        partials.push_back(builder.find_new_name("__streamk_partial"));
        builder.add_container(partials.back(), sdfg.type(reductions[f].container));
    }
    auto& peers = builder.add_for(
        winner, peer, sym::Lt(peer, sym::integer(splits)), sym::zero(), sym::add(peer, sym::one()), debug_info
    );
    auto& load_block = builder.add_block(peers.root(), debug_info);
    auto& add_block = builder.add_block(peers.root(), debug_info);
    auto peer_block = sym::add(sym::mul(tile, sym::integer(splits)), peer);
    for (int64_t f = 0; f < num_fragments; ++f) {
        auto& load = builder.add_library_node<gpu::GpuMmaFragmentLoadNode>(
            load_block, debug_info, block_size, gpu::MmaFragmentType::C, slot(peer_block, f), acc_type, impl_type
        );
        connect(load_block, partials[f], load, gpu::GpuMmaFragmentLoadNode::FRAG_INPUT_IDX, acc_pointer);
        connect(load_block, ws_name, load, gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX, ws_type);

        auto& add = builder.add_library_node<
            gpu::GpuMmaFragmentEltwiseAddNode>(add_block, debug_info, block_size, acc_type, acc_type, impl_type);
        connect(add_block, partials[f], add, gpu::GpuMmaFragmentEltwiseAddNode::FRAG_C_INPUT_IDX, acc_pointer);
        connect(
            add_block, reductions[f].container, add, gpu::GpuMmaFragmentEltwiseAddNode::FRAG_ACC_INPUT_IDX, acc_pointer
        );
        connect(add_block, reductions[f].container, add, gpu::GpuMmaFragmentEltwiseAddNode::FRAG_D_INPUT_IDX, acc_pointer);
    }

    const size_t epilogue = parent.index(election) + 1;
    while (parent.size() > epilogue) {
        builder.move_child(parent, epilogue, winner);
    }
}

void StreamK::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["num_blocks"] = num_blocks_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], grid_loop_);
}

StreamK StreamK::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto* element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    size_t num_blocks = 336;
    if (j.contains("parameters")) {
        if (j["parameters"].contains("num_blocks")) {
            num_blocks = j["parameters"]["num_blocks"].get<size_t>();
        }
    }
    return StreamK(*loop, num_blocks);
}

} // namespace transformations
} // namespace sdfg
