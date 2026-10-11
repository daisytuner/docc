#include "sdfg/tiles/transformations/software_pipelining.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/tiles/tile_target_registry.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/for_each.h"

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

// The TileCopyNode in @p block, or nullptr.
tiles::TileCopyNode* tile_copy_node_in(structured_control_flow::Block& block) {
    for (auto& node : block.dataflow().nodes()) {
        if (auto* n = dynamic_cast<tiles::TileCopyNode*>(&node)) {
            return n;
        }
    }
    return nullptr;
}

// True if @p acc's container is written in @p block: either an ordinary in-edge
// (a tasklet/copy writes it) or the access feeds a TileCopyNode's `_dst` pointer
// (the node writes through it, so the pointer is an out-edge, not an in-edge).
bool access_is_written(data_flow::DataFlowGraph& df, data_flow::AccessNode& acc) {
    if (df.in_degree(acc) > 0) {
        return true;
    }
    for (auto& e : df.out_edges(acc)) {
        if (dynamic_cast<tiles::TileCopyNode*>(&e.dst()) != nullptr && e.dst_conn() == "_dst") {
            return true;
        }
    }
    return false;
}

// Element count of a (possibly nested-array) buffer type — the per-stage stride
// once a [stages] dimension is prepended. Counts padding, so a Padded buffer's
// consecutive stages are biased by the true memory stride, not the logical tile.
symbolic::Expression buffer_element_count(const types::IType& type) {
    symbolic::Expression prod = symbolic::integer(1);
    const types::IType* cur = &type;
    while (auto* arr = dynamic_cast<const types::Array*>(cur)) {
        prod = symbolic::mul(prod, arr->num_elements());
        cur = &arr->element_type();
    }
    return prod;
}

// True if any access node in the block writes to a shared container.
bool block_writes_shared(const Function& sdfg, structured_control_flow::Block& block) {
    auto& df = block.dataflow();
    for (auto& node : df.nodes()) {
        auto* acc = dynamic_cast<data_flow::AccessNode*>(&node);
        if (acc == nullptr || !is_shared_container(sdfg, acc->data())) {
            continue;
        }
        if (access_is_written(df, *acc)) {
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
        if (access_is_written(df, *acc)) {
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

// The layout through which library node @p lib addresses its operand on memlet @p m, or nullptr.
std::optional<math::tensor::TensorLayout>
library_operand_layout(const data_flow::LibraryNode& lib, const data_flow::Memlet& m) {
    auto meta = lib.pointer_access_type(m);
    if (!meta) {
        return std::nullopt;
    }
    auto read = meta->access_read_pattern();
    auto write = meta->access_write_pattern();
    if (read && read->layout()) {
        return *read->layout();
    }
    if (write && write->layout()) {
        return *write->layout();
    }
    return std::nullopt;
}

int library_input_index(const data_flow::LibraryNode& lib, const data_flow::Memlet& m) {
    const auto& inputs = lib.inputs();
    auto it = std::find(inputs.begin(), inputs.end(), m.dst_conn());
    return it == inputs.end() ? -1 : static_cast<int>(it - inputs.begin());
}

// Library-node consumers address a staged buffer through their own operand layout
// (a bare-pointer memlet), so stage selection must be expressible as an offset bias.
bool library_consumers_relocalizable(structured_control_flow::ControlFlowNode& root, const Function& sdfg) {
    bool ok = true;
    visitor::for_each_block(root, [&](structured_control_flow::Block& b) {
        auto& dfg = b.dataflow();
        for (auto* acc : dfg.data_nodes()) {
            if (!is_shared_container(sdfg, acc->data())) {
                continue;
            }
            for (auto& m : dfg.out_edges(*acc)) {
                auto* lib = dynamic_cast<const data_flow::LibraryNode*>(&m.dst());
                if (lib == nullptr || dynamic_cast<const tiles::TileCopyNode*>(lib) != nullptr) {
                    continue;
                }
                auto layout = library_operand_layout(*lib, m);
                int idx = library_input_index(*lib, m);
                if (!layout || idx < 0 || !lib->can_relocalize_operand(idx, *layout)) {
                    ok = false;
                }
            }
        }
    });
    return ok;
}

bool is_barrier_block(structured_control_flow::ControlFlowNode& node) {
    auto* block = dynamic_cast<structured_control_flow::Block*>(&node);
    if (block == nullptr || block->dataflow().nodes().size() != 1) {
        return false;
    }
    auto libs = block->dataflow().library_nodes();
    return libs.size() == 1 && dynamic_cast<const data_flow::BarrierLocalNode*>(*libs.begin()) != nullptr;
}

// A block holding one full copy-in TileCopyNode that writes a shared container.
tiles::TileCopyNode* staging_copy(const Function& sdfg, structured_control_flow::ControlFlowNode& node) {
    auto* block = dynamic_cast<structured_control_flow::Block*>(&node);
    if (block == nullptr) {
        return nullptr;
    }
    auto* tc = tile_copy_node_in(*block);
    if (tc == nullptr || tc->direction() != tiles::CopyDirection::In || tc->phase() != tiles::CopyPhase::Full) {
        return nullptr;
    }
    for (auto& m : block->dataflow().in_edges(*tc)) {
        auto* acc = dynamic_cast<const data_flow::AccessNode*>(&m.src());
        if (m.dst_conn() == "_dst" && acc != nullptr && is_shared_container(sdfg, acc->data())) {
            return tc;
        }
    }
    return nullptr;
}

/// `[barrier; staging copies...; barrier]` in @p body: indices of the leading barrier and
/// the trailing barrier (copies lie strictly between).
struct StagingGroup {
    size_t lead;
    size_t trail;
};

std::optional<StagingGroup> find_staging_group(const Function& sdfg, structured_control_flow::Sequence& body) {
    for (size_t i = 1; i < body.size(); ++i) {
        if (staging_copy(sdfg, body.at(i)) == nullptr) {
            continue;
        }
        if (!is_barrier_block(body.at(i - 1))) {
            return std::nullopt;
        }
        size_t j = i;
        while (j + 1 < body.size() && staging_copy(sdfg, body.at(j + 1)) != nullptr) {
            ++j;
        }
        if (j + 1 >= body.size() || !is_barrier_block(body.at(j + 1))) {
            return std::nullopt;
        }
        return StagingGroup{i - 1, j + 1};
    }
    return std::nullopt;
}

// Per-thread 32-bit words a staged copy's register array needs, covering the scalar
// (one word per element), 16-byte vector and 4x4-transposing lowerings of its plan.
std::optional<size_t> stage_words(const tiles::TileCopyNode& tc, size_t elem_bytes) {
    auto total = tc.plan().src.total_elements();
    const auto& ct = tc.coop_threads();
    if (ct.is_null() || !SymEngine::is_a<SymEngine::Integer>(*total) || !SymEngine::is_a<SymEngine::Integer>(*ct)) {
        return std::nullopt;
    }
    const long long n = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
    const long long t = SymEngine::rcp_static_cast<const SymEngine::Integer>(ct)->as_int();
    if (n <= 0 || t <= 0 || elem_bytes == 0 || elem_bytes > 4) {
        return std::nullopt;
    }
    auto ceil_div = [](long long a, long long b) {
        return (a + b - 1) / b;
    };
    const long long scalar = ceil_div(n, t);
    const long long vec = 4 * ceil_div(n * static_cast<long long>(elem_bytes), 16 * t);
    const long long transpose = 8 * ceil_div(n, 16 * t);
    return static_cast<size_t>(std::max({scalar, vec, transpose}));
}

/// Element width of a staged copy, from its source pointer (the transfer width may
/// already be widened).
size_t staged_element_bytes(const tiles::TileCopyNode& tc) {
    for (auto& m : tc.get_parent().in_edges(tc)) {
        if (m.dst_conn() == "_src") {
            return types::bit_width(m.base_type().primitive_type()) / 8;
        }
    }
    return 0;
}

// The innermost sequence of @p loop's body (unwrapping single-child sequences).
structured_control_flow::Sequence& loop_body(structured_control_flow::StructuredLoop& loop) {
    structured_control_flow::Sequence* body = &loop.root();
    while (body->size() == 1) {
        auto* inner = dynamic_cast<structured_control_flow::Sequence*>(&body->at(0));
        if (inner == nullptr) {
            break;
        }
        body = inner;
    }
    return *body;
}

} // namespace

// Register-staged rewrite (see the header): prologue copy, then per panel a guarded load of
// the next panel into per-thread registers, the compute, and a guarded store into the
// (single) shared buffer between two barriers.
static void apply_register_staged(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::StructuredLoop& loop,
    const types::StorageType& register_storage
) {
    auto& sdfg = builder.subject();
    auto& body = loop_body(loop);
    auto group = find_staging_group(sdfg, body).value();
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(loop.get_parent());
    if (parent == nullptr) {
        throw InvalidTransformationException("SoftwarePipelining: panel loop must sit in a sequence");
    }
    const auto indvar = loop.indvar();
    const auto next = symbolic::add(indvar, loop.stride());
    const auto guard = symbolic::Lt(next, loop.canonical_bound());

    struct Staged {
        tiles::TileCopyNode* copy;
        std::string buffer, source;
        std::unique_ptr<types::IType> buffer_type, source_type;
    };
    std::vector<Staged> staged;
    std::vector<structured_control_flow::ControlFlowNode*> replaced;
    for (size_t i = group.lead; i <= group.trail; ++i) {
        replaced.push_back(&body.at(i));
        if (i == group.lead || i == group.trail) {
            continue;
        }
        Staged s{staging_copy(sdfg, body.at(i))};
        for (auto& m : s.copy->get_parent().in_edges(*s.copy)) {
            auto& acc = static_cast<const data_flow::AccessNode&>(m.src());
            if (m.dst_conn() == "_dst") {
                s.buffer = acc.data();
                s.buffer_type = m.base_type().clone();
            } else {
                s.source = acc.data();
                s.source_type = m.base_type().clone();
            }
        }
        staged.push_back(std::move(s));
    }

    auto add_barrier = [&](structured_control_flow::Sequence& seq) {
        builder.add_library_node<data_flow::BarrierLocalNode>(builder.add_block(seq, loop.debug_info()), DebugInfo());
    };

    // Prologue: the first panel, staged synchronously before the loop.
    auto& prologue = builder.add_sequence_before(*parent, loop, loop.debug_info());
    for (size_t i = group.lead + 1; i < group.trail; ++i) {
        deepcopy::StructuredSDFGDeepCopy dc(builder, prologue, body.at(i));
        auto mapping = dc.copy();
        const_cast<structured_control_flow::ControlFlowNode*>(mapping.at(&body.at(i)))->replace(indvar, loop.init());
    }
    add_barrier(prologue);

    auto& load_if = builder.add_if_else_before(body, body.at(group.lead), loop.debug_info());
    auto& loads = builder.add_case(load_if, guard, loop.debug_info());
    add_barrier(body);
    auto& store_if = builder.add_if_else(body, loop.debug_info());
    auto& stores = builder.add_case(store_if, guard, loop.debug_info());
    add_barrier(body);

    const types::Pointer word_ptr{types::Scalar(types::PrimitiveType::UInt32)};
    for (auto& s : staged) {
        const auto words = stage_words(*s.copy, staged_element_bytes(*s.copy)).value();
        auto reg = builder.find_new_name("__daisy_stage_" + s.buffer);
        builder.add_container(
            reg,
            types::Array(register_storage, 0, "", types::Scalar(types::PrimitiveType::UInt32), symbolic::integer(words))
        );
        auto emit_phase = [&](structured_control_flow::Sequence& seq,
                              tiles::CopyPhase phase,
                              const std::string& dst,
                              const types::IType& dst_type,
                              const std::string& src,
                              const types::IType& src_type) {
            auto& block = builder.add_block(seq, loop.debug_info());
            auto& dst_acc = builder.add_access(block, dst);
            auto& src_acc = builder.add_access(block, src);
            auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
                block,
                loop.debug_info(),
                s.copy->implementation_type(),
                s.copy->plan(),
                tiles::CopyDirection::In,
                s.copy->bytes(),
                s.copy->guard(),
                s.copy->coop_axes(),
                s.copy->coop_threads(),
                s.copy->coop_lanes()
            ));
            node.set_phase(phase);
            node.replace(indvar, next);
            builder.add_computational_memlet(block, dst_acc, node, "_dst", {}, dst_type);
            builder.add_computational_memlet(block, src_acc, node, "_src", {}, src_type);
        };
        emit_phase(loads, tiles::CopyPhase::LoadRegs, reg, word_ptr, s.source, *s.source_type);
        emit_phase(stores, tiles::CopyPhase::StoreRegs, s.buffer, *s.buffer_type, reg, word_ptr);
    }

    for (auto it = replaced.rbegin(); it != replaced.rend(); ++it) {
        builder.remove_child(body, body.index(**it));
    }
}

SoftwarePipelining::SoftwarePipelining(
    structured_control_flow::StructuredLoop& loop, size_t stages, bool single_operand, bool register_staged
)
    : loop_(loop), stages_(stages), single_operand_(single_operand), register_staged_(register_staged) {
}

std::string SoftwarePipelining::name() const {
    return "SoftwarePipelining";
}

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
    if (register_staged_) {
        // One staging group, nothing else writes shared, and every staged copy has a
        // constant per-thread share to hold in registers.
        auto& body = loop_body(loop_);
        auto group = find_staging_group(sdfg, body);
        if (!group) {
            return false;
        }
        for (size_t i = 0; i < body.size(); ++i) {
            if (i > group->lead && i < group->trail) {
                auto* tc = staging_copy(sdfg, body.at(i));
                if (!tc->guard().trivial() || tc->atom() == tiles::CopyAtom::CpAsync ||
                    !stage_words(*tc, staged_element_bytes(*tc))) {
                    return false;
                }
            } else if (i != group->lead && i != group->trail && subtree_writes_shared(sdfg, body.at(i))) {
                return false;
            }
        }
        return true;
    }
    if (!library_consumers_relocalizable(loop_.root(), sdfg)) {
        return false;
    }

    return true;
}

void SoftwarePipelining::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // The pipeline/copy nodes are stamped with the enclosing GPU tile target's
    // implementation type (CUDA/ROCm) so codegen picks that backend's dispatcher.
    data_flow::ImplementationType impl = data_flow::ImplementationType_NONE;
    const tiles::TileTarget* target = nullptr;
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (map != nullptr && tiles::AxisSchedule::classify_level(map->schedule_type()).has_value()) {
            impl = tiles::TileTargetRegistry::instance().implementation_type(map->schedule_type().value());
            target = tiles::TileTargetRegistry::instance().get(map->schedule_type().value());
            break;
        }
    }

    if (register_staged_) {
        if (target == nullptr) {
            throw InvalidTransformationException("SoftwarePipelining: no tile target for register staging");
        }
        apply_register_staged(builder, loop_, target->storage_type(tiles::Space::Register));
        analysis_manager.invalidate_all();
        return;
    }

    // Stage slot for panel p: mod((indvar - init) / stride, stages).
    auto panel = symbolic::div(symbolic::sub(loop_.indvar(), loop_.init()), loop_.stride());
    auto stage_idx = symbolic::mod(panel, symbolic::integer(stages_));

    // Collect the shared buffers the loop cooperatively stages.
    std::set<std::string> buffers;
    visitor::for_each_block(loop_.root(), [&](structured_control_flow::Block& b) {
        for (auto* acc : b.dataflow().data_nodes()) {
            if (is_shared_container(sdfg, acc->data()) && access_is_written(b.dataflow(), *acc)) {
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
        // A TileCopyNode addresses the buffer through its plan (its `_dst` memlet is a
        // bare pointer with no subset), so double-buffering biases the node's plan
        // offset by stage_idx * per-stage buffer stride (padding included) instead of
        // reindexing that memlet.
        const auto stage_stride = buffer_element_count(sdfg.type(name));
        const auto stage_bias = symbolic::mul(stage_idx, stage_stride);
        std::vector<tiles::TileCopyNode*> nodes_to_bias;
        visitor::for_each_block(loop_.root(), [&](structured_control_flow::Block& b) {
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
                    if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&m.dst())) {
                        nodes_to_bias.push_back(tc); // node writes via plan, not this memlet
                    } else if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(&m.dst())) {
                        auto layout = library_operand_layout(*lib, m);
                        int idx = library_input_index(*lib, m);
                        if (!layout || idx < 0 ||
                            !lib->relocalize_operand(
                                idx,
                                math::tensor::TensorLayout(
                                    layout->shape(), layout->strides(), symbolic::add(layout->offset(), stage_bias)
                                )
                            )) {
                            throw InvalidTransformationException(
                                "SoftwarePipelining: library node cannot address the staged buffer"
                            );
                        }
                    } else {
                        reindex(m);
                    }
                }
                for (auto& m : dfg.in_edges(*acc)) {
                    reindex(m);
                }
            }
        });
        for (auto* tc : nodes_to_bias) {
            auto plan = tc->plan();
            auto biased = symbolic::add(plan.dst.offset(), stage_bias);
            plan.dst = tiles::Layout(plan.dst.shape(), plan.dst.strides(), biased);
            tc->set_plan(plan);
        }
        builder.change_type(name, *staged);
    }

    analysis_manager.invalidate_all();

    // ---- Step 2: prologue peel + in-loop source shift + guard -------------
    // Keeping the leading/trailing barriers in place, shift each cooperative
    // copy to prefetch panel p+(stages-1) into buf[(p+stages-1)%stages], and
    // clone a prologue that fills buf[0..stages-2] with panels 0..stages-2.
    // Correct software prefetch (still synchronous; step 3 makes it cp.async).
    structured_control_flow::Sequence* body_ptr = &loop_.root();
    while (body_ptr->size() == 1) {
        auto* inner = dynamic_cast<structured_control_flow::Sequence*>(&body_ptr->at(0));
        if (inner == nullptr) {
            break;
        }
        body_ptr = inner;
    }
    auto& body = *body_ptr;
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
        builder.add_library_node<tiles::PipelineCommitNode>(commitb, loop_.debug_info(), impl);
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
    builder.add_library_node<tiles::PipelineCommitNode>(commitb, loop_.debug_info(), impl);
    auto& waitb = builder.add_block(then_branch, loop_.debug_info());
    auto& wait_node =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(waitb, loop_.debug_info(), impl, stages_ - 1));

    auto& drainb = builder.add_block(else_branch, loop_.debug_info());
    auto& drain_wait_node =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(drainb, loop_.debug_info(), impl, 0));

    // ---- Step 3: convert the synchronous copies to cp.async ----------------
    // A whole-copy TileCopyNode switches its atom to CpAsync (its dispatcher emits
    // the async transfer, degrading to a synchronous copy on non-CDNA targets).
    std::vector<structured_control_flow::Block*> copy_blocks;
    auto collect = [&](structured_control_flow::Block& b) {
        if (block_writes_any(b, pipelined)) {
            copy_blocks.push_back(&b);
        }
    };
    visitor::for_each_block(prologue, collect);
    visitor::for_each_block(body, collect);
    for (auto* b : copy_blocks) {
        if (auto* tc = tile_copy_node_in(*b)) {
            tc->set_atom(tiles::CopyAtom::CpAsync);
        }
    }

    // CUDA counts commit groups, but CDNA waits on the flat vmcnt counter, where
    // one stage expands to (sum of cp.async bytes / 4) individual global->LDS
    // loads per lane. Record that per-stage word count so the ROCm/CDNA wait can
    // emit vmcnt(keep_outstanding * loads_per_group). One loop iteration prefetches
    // exactly one stage, so summing the body's cp.async widths gives the group
    // size (a coverage loop that runs >1x per lane only makes this an under-count,
    // which over-waits — safe, never early).
    size_t loads_per_group = 0;
    visitor::for_each_block(body, [&](structured_control_flow::Block& b) {
        for (auto& node : b.dataflow().nodes()) {
            if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&node)) {
                if (tc->atom() == tiles::CopyAtom::CpAsync) {
                    loads_per_group += tc->bytes() / 4;
                }
            }
        }
    });
    if (loads_per_group > 0) {
        wait_node.set_loads_per_group(loads_per_group);
        drain_wait_node.set_loads_per_group(loads_per_group);
    }

    analysis_manager.invalidate_all();
}

void SoftwarePipelining::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["stages"] = stages_;
    j["parameters"]["single_operand"] = single_operand_;
    j["parameters"]["register_staged"] = register_staged_;

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
    bool register_staged = false;
    if (j.contains("parameters")) {
        if (j["parameters"].contains("stages")) {
            stages = j["parameters"]["stages"].get<size_t>();
        }
        if (j["parameters"].contains("single_operand")) {
            single_operand = j["parameters"]["single_operand"].get<bool>();
        }
        if (j["parameters"].contains("register_staged")) {
            register_staged = j["parameters"]["register_staged"].get<bool>();
        }
    }
    return SoftwarePipelining(*loop, stages, single_operand, register_staged);
}

} // namespace transformations
} // namespace sdfg
