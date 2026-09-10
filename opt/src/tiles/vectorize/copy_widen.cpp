#include "sdfg/tiles/vectorize/copy_widen.h"

#include <optional>
#include <string>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/tiles/library_nodes/async_copy_node.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/mul.h>

namespace sdfg {
namespace tiles {

namespace {

// The innermost array extent (row stride in elements) of a nested-array type,
// or 0 if the leaf isn't reached through arrays.
size_t innermost_array_extent(const types::IType& type) {
    const types::IType* cur = &type;
    size_t extent = 0;
    while (auto* arr = dynamic_cast<const types::Array*>(cur)) {
        auto* n = dynamic_cast<const SymEngine::Integer*>(arr->num_elements().get());
        extent = (n != nullptr) ? static_cast<size_t>(n->as_int()) : 0;
        cur = &arr->element_type();
    }
    return extent;
}

// The nearest GPU-thread-scheduled Map enclosing @p block (the cooperative
// copy's coverage map), or null.
structured_control_flow::Map* enclosing_thread_map(structured_control_flow::Block& block) {
    structured_control_flow::ControlFlowNode* n = block.get_parent();
    while (n != nullptr) {
        if (auto* m = dynamic_cast<structured_control_flow::Map*>(n)) {
            if (tiles::AxisSchedule::classify_level(m->schedule_type()).has_value()) {
                return m;
            }
        }
        n = n->get_parent();
    }
    return nullptr;
}

// Linear stride of @p e in @p coop over a @p run-length, @p run-aligned run:
//   e(coop+j) - e(coop) == stride * j  for j in [0,run) when coop % run == 0.
// Returns nullopt when it cannot be proven (any opaque use of coop). Only bare
// coop and idiv/imod(coop, M) with M a multiple of @p run are contiguity-safe: a
// run-aligned run then stays inside one M-block, so imod is unit-stride and idiv
// is constant across the run.
std::optional<long long> coop_run_stride(const symbolic::Expression& e, const symbolic::Symbol& coop, long long run) {
    if (!symbolic::uses(e, coop)) {
        return 0;
    }
    if (SymEngine::eq(*e, *coop)) {
        return 1;
    }
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        long long sum = 0;
        for (const auto& t : e->get_args()) {
            auto st = coop_run_stride(t, coop, run);
            if (!st) {
                return std::nullopt;
            }
            sum += *st;
        }
        return sum;
    }
    if (SymEngine::is_a<SymEngine::Mul>(*e)) {
        long long coeff = 1;
        symbolic::Expression var = SymEngine::null;
        for (const auto& f : e->get_args()) {
            if (symbolic::uses(f, coop)) {
                if (!var.is_null()) {
                    return std::nullopt; // coop in two factors -> nonlinear
                }
                var = f;
            } else if (SymEngine::is_a<SymEngine::Integer>(*f)) {
                coeff *= SymEngine::rcp_static_cast<const SymEngine::Integer>(f)->as_int();
            } else {
                return std::nullopt; // non-integer coefficient
            }
        }
        if (var.is_null()) {
            return 0;
        }
        auto sv = coop_run_stride(var, coop, run);
        if (!sv) {
            return std::nullopt;
        }
        return coeff * (*sv);
    }
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*e)) {
        auto fs = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(e);
        const auto& args = fs->get_args();
        const std::string name = fs->get_name();
        if (args.size() == 2 && (name == "imod" || name == "idiv")) {
            if (!SymEngine::eq(*args[0], *coop) || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
                return std::nullopt;
            }
            long long m = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
            if (m % run != 0) {
                return std::nullopt; // run may cross the M-block boundary
            }
            return (name == "imod") ? 1 : 0;
        }
    }
    return std::nullopt;
}

// The flattened run-stride of a memlet subset in @p coop over a @p run-length run:
// the innermost index carries the run, and every outer index must be provably
// constant across it (run-stride 0) — e.g. a row `idiv(coop, M)` when M is a
// multiple of @p run, so the run stays inside one row. Returns nullopt if
// unprovable.
std::optional<long long> subset_run_stride(const data_flow::Subset& subset, const symbolic::Symbol& coop, long long run) {
    if (subset.empty()) {
        return std::nullopt;
    }
    for (size_t i = 0; i + 1 < subset.size(); i++) {
        auto s = coop_run_stride(subset[i], coop, run);
        if (!s || *s != 0) {
            return std::nullopt;
        }
    }
    return coop_run_stride(subset.back(), coop, run);
}

// Stride the coop map to an absolute `factor` and validate that a run of `factor`
// elements is contiguous and aligned. Works whether the map is currently unit-
// stride (a fresh scalar copy) or already strided by a smaller factor (re-widening
// an async copy): it validates against the absolute tile coverage
// (num_iterations * current_stride) and re-strides to `factor`. Returns the widened
// byte width, or nullopt if the widen is illegal.
std::optional<size_t> try_widen_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Map* cmap,
    const data_flow::Subset& dst_subset,
    const data_flow::Subset& src_subset,
    size_t row,
    size_t elem_bytes,
    size_t factor
) {
    const size_t width = factor * elem_bytes;
    if (width != 4 && width != 8 && width != 16) {
        return std::nullopt;
    }
    if (cmap == nullptr || cmap->stride().is_null()) {
        return std::nullopt;
    }
    const long long cur_stride = cmap->stride()->as_int();
    // The target must be a further coalescing of the current per-lane step.
    if (cur_stride < 1 || factor % static_cast<size_t>(cur_stride) != 0 || row % factor != 0) {
        return std::nullopt;
    }
    auto coop = cmap->indvar();
    auto dst_stride = subset_run_stride(dst_subset, coop, static_cast<long long>(factor));
    auto src_stride = subset_run_stride(src_subset, coop, static_cast<long long>(factor));
    auto trip = cmap->num_iterations();
    auto* n = trip.is_null() ? nullptr : dynamic_cast<const SymEngine::Integer*>(trip.get());
    auto* init_i = dynamic_cast<const SymEngine::Integer*>(cmap->init().get());
    if (!(dst_stride.has_value() && *dst_stride == 1 && src_stride.has_value() && *src_stride == 1 && n != nullptr &&
          init_i != nullptr)) {
        return std::nullopt;
    }
    const long long total_tile = n->as_int() * cur_stride;
    if (total_tile % static_cast<long long>(factor) != 0 || init_i->as_int() % static_cast<long long>(factor) != 0) {
        return std::nullopt;
    }
    builder.update_loop(
        *cmap, coop, cmap->condition(), cmap->init(), symbolic::add(coop, symbolic::integer(static_cast<int>(factor)))
    );
    return width;
}

// Pick the widest legal transfer for @p elem_bytes and stride the map to it. fp32
// coalesces to float4 (16B) then float2 (8B) when @p allow_vectorize; narrow
// elements (<4B) coalesce to reach a legal >=4-byte width. Returns the resulting
// width (== elem_bytes when nothing widened).
size_t widen_copy_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Map* cmap,
    const data_flow::Subset& dst_subset,
    const data_flow::Subset& src_subset,
    size_t row,
    size_t elem_bytes,
    bool allow_vectorize
) {
    auto attempt = [&](size_t factor) -> std::optional<size_t> {
        return try_widen_map(builder, cmap, dst_subset, src_subset, row, elem_bytes, factor);
    };
    if (allow_vectorize && elem_bytes == 4) {
        if (auto w = attempt(4)) return *w;
        if (auto w = attempt(2)) return *w;
    } else if (elem_bytes > 0 && elem_bytes < 4) {
        const size_t max_factor = (allow_vectorize ? 16u : 4u) / elem_bytes;
        for (size_t width : {size_t{16}, size_t{8}, size_t{4}}) {
            const size_t factor = width / elem_bytes;
            if (width % elem_bytes == 0 && factor > 1 && factor <= max_factor) {
                if (auto w = attempt(factor)) return *w;
            }
        }
    }
    return elem_bytes;
}

} // namespace

std::optional<size_t> rewrite_cooperative_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    bool allow_vectorize,
    CopyTransfer transfer,
    const data_flow::ImplementationType& implementation_type
) {
    auto& df = block.dataflow();
    data_flow::Tasklet* tk = nullptr;
    for (auto& node : df.nodes()) {
        if (auto* t = dynamic_cast<data_flow::Tasklet*>(&node)) {
            tk = t;
            break;
        }
    }
    if (tk == nullptr) {
        return std::nullopt;
    }
    data_flow::Memlet* in_m = nullptr;
    for (auto& m : df.in_edges(*tk)) {
        in_m = &m;
        break;
    }
    data_flow::Memlet* out_m = nullptr;
    for (auto& m : df.out_edges(*tk)) {
        out_m = &m;
        break;
    }
    if (in_m == nullptr || out_m == nullptr) {
        return std::nullopt;
    }
    auto* src_acc = dynamic_cast<data_flow::AccessNode*>(&in_m->src());
    auto* dst_acc = dynamic_cast<data_flow::AccessNode*>(&out_m->dst());
    if (src_acc == nullptr || dst_acc == nullptr) {
        return std::nullopt;
    }
    const std::string src_name = src_acc->data();
    const std::string dst_name = dst_acc->data();
    const data_flow::Subset src_subset = in_m->subset();
    const data_flow::Subset dst_subset = out_m->subset();
    types::Scalar src_leaf(in_m->base_type().primitive_type());
    types::Scalar dst_leaf(out_m->base_type().primitive_type());
    types::Pointer src_ptr_t(src_leaf);
    types::Pointer dst_ptr_t(dst_leaf);
    const size_t elem_bytes = types::bit_width(out_m->base_type().primitive_type()) / 8;

    // Coalesce contiguous elements per thread into one wider transfer, striding the
    // coop map to match (all the legality/alignment checks live in widen_copy_map).
    auto* cmap = enclosing_thread_map(block);
    const size_t row = innermost_array_extent(out_m->base_type());
    const size_t bytes = widen_copy_map(builder, cmap, dst_subset, src_subset, row, elem_bytes, allow_vectorize);

    if (bytes != 4 && bytes != 8 && bytes != 16) {
        return std::nullopt; // could not reach a legal transfer width; leave scalar
    }
    // A synchronous vector copy is only worth a node when coalescing widened the
    // transfer; an un-widened element stays the original scalar tasklet.
    if (transfer == CopyTransfer::VectorSync && bytes <= elem_bytes) {
        return std::nullopt;
    }

    auto* pseq = dynamic_cast<structured_control_flow::Sequence*>(block.get_parent());
    if (pseq == nullptr) {
        return std::nullopt;
    }

    const auto src_ptr_name = builder.find_new_name("__daisy_cp_src");
    const auto dst_ptr_name = builder.find_new_name("__daisy_cp_dst");
    builder.add_container(src_ptr_name, src_ptr_t);
    builder.add_container(dst_ptr_name, dst_ptr_t);

    // Reference block: take addresses of the shared dst slot and the global src.
    // The reference base_type is the indexed container's own type (so the subset
    // dimensions match); the result access node holds a pointer-to-element.
    auto& refb = builder.add_block_before(*pseq, block, block.debug_info());
    auto& s_acc = builder.add_access(refb, src_name);
    auto& d_acc = builder.add_access(refb, dst_name);
    auto& src_ptr_w = builder.add_access(refb, src_ptr_name);
    auto& dst_ptr_w = builder.add_access(refb, dst_ptr_name);
    builder.add_reference_memlet(refb, s_acc, src_ptr_w, src_subset, in_m->base_type());
    builder.add_reference_memlet(refb, d_acc, dst_ptr_w, dst_subset, out_m->base_type());

    // Node block: the widened transfer from the global src ptr into the dst ptr.
    auto& nodeb = builder.add_block_before(*pseq, block, block.debug_info());
    auto& src_ptr_r = builder.add_access(nodeb, src_ptr_name);
    auto& dst_ptr_r = builder.add_access(nodeb, dst_ptr_name);
    data_flow::LibraryNode* node = nullptr;
    if (transfer == CopyTransfer::CpAsync) {
        node = &builder.add_library_node<tiles::CpAsyncCopyNode>(nodeb, block.debug_info(), implementation_type, bytes);
    } else {
        node = &builder.add_library_node<tiles::VectorCopyNode>(nodeb, block.debug_info(), implementation_type, bytes);
    }
    builder.add_computational_memlet(nodeb, dst_ptr_r, *node, "_dst", {}, dst_ptr_t);
    builder.add_computational_memlet(nodeb, src_ptr_r, *node, "_src", {}, src_ptr_t);

    builder.remove_child(*pseq, pseq->index(block));
    return bytes;
}

std::optional<size_t> widen_existing_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& node_block,
    data_flow::LibraryNode& node,
    size_t current_bytes
) {
    // The node's {_dst,_src} come from the two pointer containers written by the
    // preceding reference block; recover the buffer/global subsets + element type
    // from those reference memlets to re-run the contiguity analysis.
    auto& ndf = node_block.dataflow();
    std::string dst_ptr, src_ptr;
    for (auto& e : ndf.in_edges(node)) {
        auto* a = dynamic_cast<data_flow::AccessNode*>(&e.src());
        if (a == nullptr) {
            continue;
        }
        if (e.dst_conn() == "_dst") {
            dst_ptr = a->data();
        } else if (e.dst_conn() == "_src") {
            src_ptr = a->data();
        }
    }
    if (dst_ptr.empty() || src_ptr.empty()) {
        return std::nullopt;
    }

    auto* pseq = dynamic_cast<structured_control_flow::Sequence*>(node_block.get_parent());
    if (pseq == nullptr) {
        return std::nullopt;
    }
    size_t idx = pseq->index(node_block);
    if (idx == 0) {
        return std::nullopt;
    }
    auto* ref_block = dynamic_cast<structured_control_flow::Block*>(&pseq->at(idx - 1));
    if (ref_block == nullptr) {
        return std::nullopt;
    }

    data_flow::Subset dst_subset, src_subset;
    const types::IType* dst_bt = nullptr;
    for (auto* a : ref_block->dataflow().data_nodes()) {
        for (auto& m : ref_block->dataflow().in_edges(*a)) {
            if (a->data() == dst_ptr) {
                dst_subset = m.subset();
                dst_bt = &m.base_type();
            } else if (a->data() == src_ptr) {
                src_subset = m.subset();
            }
        }
    }
    if (dst_bt == nullptr) {
        return std::nullopt;
    }

    const size_t elem_bytes = types::bit_width(dst_bt->primitive_type()) / 8;
    if (elem_bytes == 0) {
        return std::nullopt;
    }
    auto* cmap = enclosing_thread_map(node_block);
    const size_t row = innermost_array_extent(*dst_bt);
    const size_t new_bytes =
        widen_copy_map(builder, cmap, dst_subset, src_subset, row, elem_bytes, /*allow_vectorize=*/true);
    if (new_bytes <= current_bytes) {
        return std::nullopt;
    }
    if (auto* cp = dynamic_cast<tiles::CpAsyncCopyNode*>(&node)) {
        cp->set_bytes(new_bytes);
    } else if (auto* vc = dynamic_cast<tiles::VectorCopyNode*>(&node)) {
        vc->set_bytes(new_bytes);
    } else {
        return std::nullopt;
    }
    return new_bytes;
}

} // namespace tiles
} // namespace sdfg
