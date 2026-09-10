#include "sdfg/tiles/transformations/software_pipelining.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "sdfg/data_flow/access_node.h"
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
#include "sdfg/tiles/library_nodes/async_copy_node.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/tiles/tile_target_registry.h"
#include "sdfg/tiles/vectorize/copy_widen.h"
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

} // namespace

SoftwarePipelining::SoftwarePipelining(structured_control_flow::StructuredLoop& loop, size_t stages, bool single_operand)
    : loop_(loop), stages_(stages), single_operand_(single_operand) {}

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

    // The pipeline/copy nodes are stamped with the enclosing GPU tile target's
    // implementation type (CUDA/ROCm) so codegen picks that backend's dispatcher.
    data_flow::ImplementationType impl = data_flow::ImplementationType_NONE;
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (map != nullptr && tiles::AxisSchedule::classify_level(map->schedule_type()).has_value()) {
            impl = tiles::TileTargetRegistry::instance().implementation_type(map->schedule_type().value());
            break;
        }
    }

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
    auto& drain_wait_node = static_cast<
        tiles::PipelineWaitNode&>(builder.add_library_node<tiles::PipelineWaitNode>(drainb, loop_.debug_info(), impl, 0)
    );

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
        // Minimal legal cp.async (narrow types coalesce to 4 bytes); TileVectorizer
        // widens further for performance.
        tiles::rewrite_cooperative_copy(builder, *b, /*allow_vectorize=*/false, tiles::CopyTransfer::CpAsync, impl);
    }

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
            if (auto* cp = dynamic_cast<tiles::CpAsyncCopyNode*>(&node)) {
                loads_per_group += cp->bytes() / 4;
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
    if (j.contains("parameters")) {
        if (j["parameters"].contains("stages")) {
            stages = j["parameters"]["stages"].get<size_t>();
        }
        if (j["parameters"].contains("single_operand")) {
            single_operand = j["parameters"]["single_operand"].get<bool>();
        }
    }
    return SoftwarePipelining(*loop, stages, single_operand);
}

} // namespace transformations
} // namespace sdfg
