#include "sdfg/tiles/transformations/software_pipelining.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/async_copy_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/mul.h>

namespace sdfg {
namespace transformations {

namespace {

// A container is a block-shared buffer if its declared storage is NV_Shared.
bool is_shared_container(const Function& sdfg, const std::string& name) {
    try {
        return sdfg.type(name).storage_type().is_nv_shared();
    } catch (...) {
        return false;
    }
}

// True if any access node in the block writes to a shared container.
bool block_writes_shared(const Function& sdfg, structured_control_flow::Block& block) {
    auto& df = block.dataflow();
    for (auto& node : df.nodes()) {
        auto* acc = dynamic_cast<data_flow::AccessNode*>(&node);
        if (acc == nullptr || !is_shared_container(sdfg, acc->data())) {
            continue;
        }
        if (df.in_degree(*acc) > 0) {
            return true;
        }
    }
    return false;
}

// Recursively: does this subtree write to a shared container?
bool subtree_writes_shared(const Function& sdfg, structured_control_flow::ControlFlowNode& node) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return block_writes_shared(sdfg, *block);
    }
    if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            if (subtree_writes_shared(sdfg, seq->at(i))) {
                return true;
            }
        }
        return false;
    }
    if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
        return subtree_writes_shared(sdfg, map->root());
    }
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return subtree_writes_shared(sdfg, loop->root());
    }
    if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            if (subtree_writes_shared(sdfg, if_else->at(i).first)) {
                return true;
            }
        }
        return false;
    }
    return false;
}

// True if the block writes to one of the named containers.
bool block_writes_any(structured_control_flow::Block& block, const std::set<std::string>& names) {
    auto& df = block.dataflow();
    for (auto& node : df.nodes()) {
        auto* acc = dynamic_cast<data_flow::AccessNode*>(&node);
        if (acc == nullptr || names.count(acc->data()) == 0) {
            continue;
        }
        if (df.in_degree(*acc) > 0) {
            return true;
        }
    }
    return false;
}

// Recursively: does this subtree write to one of the named containers?
bool subtree_writes_any(structured_control_flow::ControlFlowNode& node, const std::set<std::string>& names) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return block_writes_any(*block, names);
    }
    if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            if (subtree_writes_any(seq->at(i), names)) {
                return true;
            }
        }
        return false;
    }
    if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
        return subtree_writes_any(map->root(), names);
    }
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return subtree_writes_any(loop->root(), names);
    }
    if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            if (subtree_writes_any(if_else->at(i).first, names)) {
                return true;
            }
        }
        return false;
    }
    return false;
}

// Visit every Block reachable under @p node.
void for_each_block(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::Block&)>& fn
) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        fn(*block);
    } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            for_each_block(seq->at(i), fn);
        }
    } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
        for_each_block(map->root(), fn);
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        for_each_block(loop->root(), fn);
    } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            for_each_block(if_else->at(i).first, fn);
        }
    }
}

// Prepend a leading `[stages]` axis to a nested-array buffer type, keeping the
// NV_Shared storage on the (new) outermost axis only.
std::unique_ptr<types::IType> prepend_stage_dim(const types::IType& buf, size_t stages) {
    std::vector<symbolic::Expression> dims;
    const types::IType* cur = &buf;
    while (auto* arr = dynamic_cast<const types::Array*>(cur)) {
        dims.push_back(arr->num_elements());
        cur = &arr->element_type();
    }
    std::unique_ptr<types::IType> inner = cur->clone(); // scalar element
    for (size_t a = dims.size(); a >= 1; a--) {
        inner = std::make_unique<types::Array>(*inner, dims[a - 1]);
    }
    return std::make_unique<
        types::Array>(buf.storage_type(), buf.alignment(), buf.initializer(), *inner, symbolic::integer(stages));
}

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

// Linear stride of @p e in @p coop over a length-4, 4-aligned run:
//   e(coop+j) - e(coop) == stride * j  for j in [0,3] when coop % 4 == 0.
// Returns nullopt when it cannot be proven (any opaque use of coop). Only bare
// coop and idiv/imod(coop, M) with M a multiple of 4 are contiguity-safe: a
// 4-aligned run then stays inside one M-block, so imod is unit-stride and idiv
// is constant across the run.
std::optional<long long> coop_run_stride(const symbolic::Expression& e, const symbolic::Symbol& coop) {
    if (!symbolic::uses(e, coop)) {
        return 0;
    }
    if (SymEngine::eq(*e, *coop)) {
        return 1;
    }
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        long long sum = 0;
        for (const auto& t : e->get_args()) {
            auto st = coop_run_stride(t, coop);
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
        auto sv = coop_run_stride(var, coop);
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
            if (m % 4 != 0) {
                return std::nullopt; // run may cross the M-block boundary
            }
            return (name == "imod") ? 1 : 0;
        }
    }
    return std::nullopt;
}

// The flattened run-stride of a memlet subset in @p coop: only the innermost
// index may vary with coop (outer indices multiply larger extents, so any coop
// dependence there breaks contiguity). Returns nullopt if unprovable.
std::optional<long long> subset_run_stride(const data_flow::Subset& subset, const symbolic::Symbol& coop) {
    if (subset.empty()) {
        return std::nullopt;
    }
    for (size_t i = 0; i + 1 < subset.size(); i++) {
        if (symbolic::uses(subset[i], coop)) {
            return std::nullopt;
        }
    }
    return coop_run_stride(subset.back(), coop);
}

// Rewrite a synchronous copy block (src[..] --assign--> shared_buf[..]) into a
// cp.async: reference memlets take &shared_buf[..] and &src[..], and a
// CpAsyncCopyNode streams the element directly. The replacement blocks are
// inserted before @p block, then @p block is removed. When @p vectorize is set
// and the copied run is a contiguous, 16-byte-aligned float4, the cooperative
// map is strided by 4 and a single 16-byte cp.async replaces four scalar ones.
void convert_copy_block_to_async(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block, bool vectorize
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
        return;
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
        return;
    }
    auto* src_acc = dynamic_cast<data_flow::AccessNode*>(&in_m->src());
    auto* dst_acc = dynamic_cast<data_flow::AccessNode*>(&out_m->dst());
    if (src_acc == nullptr || dst_acc == nullptr) {
        return;
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

    // cp.async only moves 4-, 8-, or 16-byte transfers. Coalesce `factor` contiguous
    // elements per thread into one transfer of `factor * elem_bytes` bytes: both the
    // shared destination and the global source must be unit-stride in the cooperative
    // indvar, the shared row `factor`-aligned, and the copy count/init a multiple of
    // `factor`. Then stride the coop map by `factor`.
    size_t bytes = elem_bytes;
    auto* cmap = enclosing_thread_map(block);
    const size_t row = innermost_array_extent(out_m->base_type());
    auto try_widen = [&](size_t factor) -> bool {
        const size_t width = factor * elem_bytes;
        if (width != 4 && width != 8 && width != 16) {
            return false;
        }
        if (cmap == nullptr || cmap->stride().is_null() || cmap->stride()->as_int() != 1 || row % factor != 0) {
            return false;
        }
        auto coop = cmap->indvar();
        auto dst_stride = subset_run_stride(dst_subset, coop);
        auto src_stride = subset_run_stride(src_subset, coop);
        auto trip = cmap->num_iterations();
        auto* n = trip.is_null() ? nullptr : dynamic_cast<const SymEngine::Integer*>(trip.get());
        auto* init_i = dynamic_cast<const SymEngine::Integer*>(cmap->init().get());
        if (!(dst_stride.has_value() && *dst_stride == 1 && src_stride.has_value() && *src_stride == 1 &&
              n != nullptr && n->as_int() % static_cast<int>(factor) == 0 && init_i != nullptr &&
              init_i->as_int() % static_cast<int>(factor) == 0)) {
            return false;
        }
        builder.update_loop(
            *cmap,
            coop,
            cmap->condition(),
            cmap->init(),
            symbolic::add(coop, symbolic::integer(static_cast<int>(factor)))
        );
        bytes = width;
        return true;
    };

    // fp32 keeps its float4 (16-byte) path, gated by `vectorize`. Narrow elements
    // (fp16/int8) must coalesce to reach a legal >=4-byte width even without
    // `vectorize`, since a scalar sub-4-byte cp.async is illegal; pick the widest
    // legal transfer (16B, then 8B, then 4B).
    if (vectorize && elem_bytes == 4) {
        try_widen(4);
    } else if (elem_bytes > 0 && elem_bytes < 4) {
        const size_t max_factor = (vectorize ? 16u : 4u) / elem_bytes;
        for (size_t width : {size_t{16}, size_t{8}, size_t{4}}) {
            const size_t factor = width / elem_bytes;
            if (width % elem_bytes == 0 && factor > 1 && factor <= max_factor && try_widen(factor)) {
                break;
            }
        }
    }

    // cp.async has no legal sub-4-byte transfer. If a narrow copy could not be
    // widened (e.g. non-contiguous), keep the synchronous copy block rather than
    // emitting an illegal cp.async.
    if (bytes != 4 && bytes != 8 && bytes != 16) {
        return;
    }

    auto* pseq = dynamic_cast<structured_control_flow::Sequence*>(block.get_parent());
    if (pseq == nullptr) {
        return;
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

    // Node block: cp.async from the global src ptr into the shared dst ptr.
    auto& nodeb = builder.add_block_before(*pseq, block, block.debug_info());
    auto& src_ptr_r = builder.add_access(nodeb, src_ptr_name);
    auto& dst_ptr_r = builder.add_access(nodeb, dst_ptr_name);
    auto& node = builder.add_library_node<data_flow::CpAsyncCopyNode>(nodeb, block.debug_info(), bytes);
    builder.add_computational_memlet(nodeb, dst_ptr_r, node, "_dst", {}, dst_ptr_t);
    builder.add_computational_memlet(nodeb, src_ptr_r, node, "_src", {}, src_ptr_t);

    builder.remove_child(*pseq, pseq->index(block));
}

} // namespace

SoftwarePipelining::SoftwarePipelining(
    structured_control_flow::StructuredLoop& loop, size_t stages, bool single_operand, bool vectorize
)
    : loop_(loop), stages_(stages), single_operand_(single_operand), vectorize_(vectorize) {}

std::string SoftwarePipelining::name() const { return "SoftwarePipelining"; }

bool SoftwarePipelining::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (stages_ < 2) {
        return false;
    }
    auto& sdfg = builder.subject();

    // A parallel loop (Map) has no cross-iteration order to pipeline over — only
    // a sequential panel loop qualifies.
    if (dynamic_cast<structured_control_flow::Map*>(&loop_) != nullptr) {
        return false;
    }

    // cp.async is a GPU primitive; require a GPU-offloaded ancestor (the block
    // context that also owns the __syncthreads the pipeline fences against).
    bool gpu_ancestor = false;
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (map != nullptr && tiles::AxisSchedule::classify_level(map->schedule_type()).has_value()) {
            gpu_ancestor = true;
            break;
        }
    }
    if (!gpu_ancestor) {
        return false;
    }

    // The panel count must be a compile-time constant >= stages (a partial pipe
    // over a runtime count would need dynamic guards on every stage). Use the
    // over-approximating count so tiled panel loops with a compound bound like
    // `k < K && k < k_chunk + T` (symbolic init) still resolve their constant
    // tile trip T/stride via the min-distribution in num_iterations_approx().
    if (loop_.canonical_bound().is_null()) {
        return false;
    }
    auto trip = loop_.num_iterations_approx();
    if (trip.is_null() || !SymEngine::is_a<SymEngine::Integer>(*trip)) {
        return false;
    }
    if (SymEngine::rcp_static_cast<const SymEngine::Integer>(trip)->as_int() < static_cast<long long>(stages_)) {
        return false;
    }

    // The body must cooperatively stage a shared tile (a copy that writes shared)
    // and then consume it — i.e. at least one shared-writing sub-scope exists.
    if (!subtree_writes_shared(sdfg, loop_.root())) {
        return false;
    }

    return true;
}

void SoftwarePipelining::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Stage slot for panel p: mod((indvar - init) / stride, stages).
    auto panel = symbolic::div(symbolic::sub(loop_.indvar(), loop_.init()), loop_.stride());
    auto stage_idx = symbolic::mod(panel, symbolic::integer(stages_));

    // Collect the shared buffers the loop cooperatively stages.
    std::set<std::string> buffers;
    for_each_block(loop_.root(), [&](structured_control_flow::Block& b) {
        for (auto* acc : b.dataflow().data_nodes()) {
            if (is_shared_container(sdfg, acc->data()) && b.dataflow().in_degree(*acc) > 0) {
                buffers.insert(acc->data());
            }
        }
    });

    // Double-buffer each: prepend a [stages] axis to the type and index it by
    // stage_idx on every memlet that touches the buffer inside the loop.
    // In single-operand mode pipeline only the first (name-ordered) buffer; the
    // rest stay single-buffered + synchronous so shared stays small enough to
    // keep occupancy.
    std::set<std::string> pipelined = buffers;
    if (single_operand_ && buffers.size() > 1) {
        pipelined = {*buffers.begin()};
    }
    for (const auto& name : pipelined) {
        auto staged = prepend_stage_dim(sdfg.type(name), stages_);
        for_each_block(loop_.root(), [&](structured_control_flow::Block& b) {
            auto& dfg = b.dataflow();
            for (auto* acc : dfg.data_nodes()) {
                if (acc->data() != name) {
                    continue;
                }
                auto reindex = [&](data_flow::Memlet& m) {
                    data_flow::Subset s = m.subset();
                    s.insert(s.begin(), stage_idx);
                    m.set_subset(s);
                    m.set_base_type(*staged);
                };
                for (auto& m : dfg.out_edges(*acc)) {
                    reindex(m);
                }
                for (auto& m : dfg.in_edges(*acc)) {
                    reindex(m);
                }
            }
        });
        builder.change_type(name, *staged);
    }

    analysis_manager.invalidate_all();

    // ---- Step 2: prologue peel + in-loop source shift + guard -------------
    // Keeping the leading/trailing barriers in place, shift each cooperative
    // copy to prefetch panel p+(stages-1) into buf[(p+stages-1)%stages], and
    // clone a prologue that fills buf[0..stages-2] with panels 0..stages-2.
    // Correct software prefetch (still synchronous; step 3 makes it cp.async).
    auto& body = loop_.root();
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(loop_.get_parent());
    if (parent == nullptr) {
        return;
    }
    auto init = loop_.init();
    auto stride = loop_.stride();
    auto indvar = loop_.indvar();
    auto bound = loop_.canonical_bound();

    // The copy sub-scopes (direct body children that write a pipelined buffer).
    std::vector<structured_control_flow::ControlFlowNode*> copies;
    for (size_t i = 0; i < body.size(); i++) {
        if (subtree_writes_any(body.at(i), pipelined)) {
            copies.push_back(&body.at(i));
        }
    }

    // Prologue (before the loop): panels 0..stages-2 into their stage slots,
    // committing each panel's copy group so they can be waited on in order.
    auto& prologue = builder.add_sequence_before(*parent, loop_, loop_.debug_info());
    for (size_t s = 0; s + 1 < stages_; s++) {
        auto panel_k = symbolic::add(init, symbolic::mul(symbolic::integer(static_cast<long long>(s)), stride));
        for (auto* copy : copies) {
            deepcopy::StructuredSDFGDeepCopy dc(builder, prologue, *copy);
            auto mapping = dc.copy();
            auto* clone = const_cast<structured_control_flow::ControlFlowNode*>(mapping.at(copy));
            clone->replace(indvar, panel_k);
        }
        auto& commitb = builder.add_block(prologue, loop_.debug_info());
        builder.add_library_node<data_flow::PipelineCommitNode>(commitb, loop_.debug_info());
    }

    // In-loop: shift each copy to panel indvar+(stages-1)*stride and guard it so
    // no out-of-range panel is prefetched on the final iterations. The prefetched
    // panel indvar+shift is valid iff it is still below the loop's own bound
    // (exact and symbolic-safe, so a compound/symbolic-init tile bound works too).
    auto shift = symbolic::mul(symbolic::integer(static_cast<long long>(stages_ - 1)), stride);
    auto guard_cond = symbolic::Lt(symbolic::add(indvar, shift), bound);

    // One if-else guards the whole prefetch+commit+wait region:
    //   if (indvar + (stages-1)*stride < bound):
    //       prefetch panel indvar+shift; commit; wait keeping stages-1 in flight
    //   else:  // final stages-1 iterations — nothing new was prefetched
    //       wait for *all* outstanding loads (keep 0) so the buffer we are about
    //       to consume is complete.
    // The else branch is essential on CDNA: its wait lowers to `s_waitcnt
    // vmcnt(keep*loads_per_group)`, and with `keep = stages-1` the only loads
    // still in flight on the tail are exactly the buffer being read, so that
    // wait would be a no-op and the last panel would read incomplete LDS.
    auto& if_else = builder.add_if_else_before(body, *copies.front());
    auto& then_branch = builder.add_case(if_else, guard_cond, loop_.debug_info());
    auto& else_branch = builder.add_case(if_else, symbolic::Not(guard_cond), loop_.debug_info());

    for (auto* copy : copies) {
        copy->replace(indvar, symbolic::add(indvar, shift));
        builder.move_child(body, body.index(*copy), then_branch);
    }

    auto& commitb = builder.add_block(then_branch, loop_.debug_info());
    builder.add_library_node<data_flow::PipelineCommitNode>(commitb, loop_.debug_info());
    auto& waitb = builder.add_block(then_branch, loop_.debug_info());
    auto& wait_node =
        static_cast<data_flow::PipelineWaitNode&>(builder.add_library_node<
                                                  data_flow::PipelineWaitNode>(waitb, loop_.debug_info(), stages_ - 1));

    auto& drainb = builder.add_block(else_branch, loop_.debug_info());
    auto& drain_wait_node =
        static_cast<data_flow::PipelineWaitNode&>(builder.add_library_node<
                                                  data_flow::PipelineWaitNode>(drainb, loop_.debug_info(), 0));

    analysis_manager.invalidate_all();

    // ---- Step 3: convert the synchronous copies to cp.async ----------------
    // Every shared-writing assign becomes a CpAsyncCopyNode (address-of src/dst
    // via reference memlets). The node degrades to a synchronous copy on ROCm.
    std::vector<structured_control_flow::Block*> copy_blocks;
    auto collect = [&](structured_control_flow::Block& b) {
        if (block_writes_any(b, pipelined)) {
            copy_blocks.push_back(&b);
        }
    };
    for_each_block(prologue, collect);
    for_each_block(body, collect);
    for (auto* b : copy_blocks) {
        convert_copy_block_to_async(builder, *b, vectorize_);
    }

    analysis_manager.invalidate_all();

    // CUDA counts commit groups, but CDNA waits on the flat vmcnt counter, where
    // one stage expands to (sum of cp.async bytes / 4) individual global->LDS
    // loads per lane. Record that per-stage word count so the ROCm/CDNA wait can
    // emit vmcnt(keep_outstanding * loads_per_group). One loop iteration prefetches
    // exactly one stage, so summing the body's CpAsyncCopyNodes gives the group
    // size (a coverage loop that runs >1x per lane only makes this an under-count,
    // which over-waits — safe, never early).
    size_t loads_per_group = 0;
    for_each_block(body, [&](structured_control_flow::Block& b) {
        for (auto& node : b.dataflow().nodes()) {
            if (auto* cp = dynamic_cast<data_flow::CpAsyncCopyNode*>(&node)) {
                loads_per_group += cp->bytes() / 4;
            }
        }
    });
    if (loads_per_group > 0) {
        wait_node.set_loads_per_group(loads_per_group);
        drain_wait_node.set_loads_per_group(loads_per_group);
    }
}

void SoftwarePipelining::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["stages"] = stages_;
    j["parameters"]["single_operand"] = single_operand_;
    j["parameters"]["vectorize"] = vectorize_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

SoftwarePipelining SoftwarePipelining::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
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
    size_t stages = 2;
    bool single_operand = false;
    bool vectorize = false;
    if (j.contains("parameters")) {
        if (j["parameters"].contains("stages")) {
            stages = j["parameters"]["stages"].get<size_t>();
        }
        if (j["parameters"].contains("single_operand")) {
            single_operand = j["parameters"]["single_operand"].get<bool>();
        }
        if (j["parameters"].contains("vectorize")) {
            vectorize = j["parameters"]["vectorize"].get<bool>();
        }
    }
    return SoftwarePipelining(*loop, stages, single_operand, vectorize);
}

} // namespace transformations
} // namespace sdfg
