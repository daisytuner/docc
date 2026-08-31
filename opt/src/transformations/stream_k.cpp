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
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/scalar.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

namespace {

using namespace structured_control_flow;

bool is_grid_level(gpu::TargetLevel level) {
    return level == gpu::TargetLevel::X_GRID || level == gpu::TargetLevel::Y_GRID || level == gpu::TargetLevel::Z_GRID;
}

// A GPU-offloaded loop whose schedule maps it to a grid axis.
bool is_grid_offloaded(const StructuredLoop& loop) {
    try {
        return is_grid_level(gpu::ScheduleType_GPU_Offload::target_level(loop.schedule_type()));
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
            if (auto* found = find_reduce(seq->at(i))) return found;
        }
    } else if (auto* loop = dynamic_cast<StructuredLoop*>(&node)) {
        return find_reduce(loop->root());
    } else if (auto* if_else = dynamic_cast<IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i) {
            if (auto* found = find_reduce(if_else->at(i).first)) return found;
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

} // namespace

StreamK::StreamK(structured_control_flow::StructuredLoop& grid_loop, size_t num_blocks)
    : grid_loop_(grid_loop), num_blocks_(num_blocks) {}

std::string StreamK::name() const { return "StreamK"; }

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

    return true;
}

void StreamK::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    namespace sym = symbolic;
    using namespace structured_control_flow;

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
        for (size_t j = i + 1; j < num_dims; ++j) divisor = sym::mul(divisor, trips[j]);
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
