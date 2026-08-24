#include "sdfg/transformations/local_storage.h"

#include <functional>
#include <unordered_set>

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/data_flow/library_node.h"
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

LocalStorage::Locality LocalStorage::derive_storage(const LocalityPlan& plan, bool container_written) {
    using Level = LocalityPlan::Level;
    // A cooperative CPU-parallel dim would need threads to share a stack — impossible.
    if (plan.has_cpu_cooperative()) {
        return Locality::Reject;
    }
    if (plan.has_gpu_cooperative()) {
        // A cooperative write across threads is a reduction: that is owned by the
        // Reduce node + reduce dispatcher, not LocalStorage.
        if (container_written) {
            return Locality::Reject;
        }
        // A cooperative buffer lives in a device scope inside the kernel, below
        // the outermost loop.
        if (!plan.inside_gpu_kernel() || plan.loop_is_outermost) {
            return Locality::Reject;
        }
        // Storage follows the coarsest cooperative level.
        if (plan.has_cooperative_at(Level::Grid)) {
            return Locality::Global;
        }
        if (plan.has_cooperative_at(Level::Block)) {
            return Locality::Shared;
        }
        // Warp-only cooperation is served by shuffles, not a staged buffer.
        return Locality::Reject;
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

    // Reduction accumulators are staged and combined by the Reduce node + reduce
    // dispatcher; LocalStorage stages read-only operands only and must not also
    // localize an accumulator.
    if (is_reduction_accumulator(loop_, container_, analysis_manager)) {
        return false;
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
            for (const auto& d : plan_.dims) {
                if (!d.is_gpu || d.level != LocalityPlan::Level::Block) {
                    return false; // no CPU / grid / warp dims in the mix
                }
            }
            auto coop_dims = plan_.gpu_cooperative_dims();
            if (coop_dims.size() != 1) {
                return false; // exactly one cooperative (copy) axis in v1
            }
            // The first loop ancestor must be the cooperative Map itself.
            structured_control_flow::Map* coop_map = nullptr;
            for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
                if (auto* enclosing = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    coop_map = dynamic_cast<structured_control_flow::Map*>(enclosing);
                    break;
                }
            }
            if (!coop_map || !symbolic::eq(coop_map->indvar(), coop_dims.front().indvar)) {
                return false;
            }
            // The cooperative copy is lowered by the new offload dispatcher, so
            // only the *_Offload schedules are supported (not the legacy ones).
            const std::string& sched_value = coop_map->schedule_type().value();
            if (sched_value != "CUDA_Offload" && sched_value != "ROCM_Offload") {
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
    if (storage_type_.is_nv_shared()) {
        for (const auto& d : plan_.gpu_per_thread_dims()) {
            slot_sizes.push_back(d.parallel_size);
            slot_indices.push_back(symbolic::mod(d.indvar, d.parallel_size));
        }
    }

    TileBuffer buffer{slot_sizes, tile_info_.varying_sizes()};
    types::Array buffer_type(storage_type_, 0, {}, scalar_type, buffer.total_size());
    builder.add_container(local_name_, buffer_type);

    if (storage_type_.is_nv_shared()) {
        // Read-only cooperative tile: cooperative copy-in + barrier(s), no writeback.
        // A per-thread slot prefix means the shared row is re-staged per kernel
        // coverage iteration, so guard it with a leading barrier too.
        bool leading_barrier = !slot_indices.empty();
        emit_cooperative_copy_in(builder, *parent, buffer, buffer_type, pointer_type, slot_indices, leading_barrier);
    } else {
        if (needs_copy_in()) {
            emit_private_copy(builder, *parent, buffer, buffer_type, pointer_type, /*writeback=*/false);
        }
        if (needs_copy_out()) {
            emit_private_copy(builder, *parent, buffer, buffer_type, pointer_type, /*writeback=*/true);
        }
    }

    rewrite_body(builder, analysis_manager, buffer, buffer_type, slot_indices);
    analysis_manager.invalidate_all();
}

void LocalStorage::emit_private_copy(
    builder::StructuredSDFGBuilder& builder,
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
        builder.add_container(name, types::Scalar(types::PrimitiveType::UInt64));
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

    auto& block = builder.add_block(*current);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    data_flow::Subset original_subset = tile_info_.original_subset(indvars);
    data_flow::Subset buffer_subset = {buffer.linearize({}, indvars)};
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

    builder.move_children(scope, parent, index + 1);
    builder.remove_child(parent, index);
}

void LocalStorage::emit_cooperative_copy_in(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const TileBuffer& buffer,
    const types::IType& buffer_type,
    const types::IType& pointer_type,
    const std::vector<symbolic::Expression>& slot_indices,
    bool leading_barrier
) {
    // The cooperative dim's Map (immediate enclosing loop; verified in
    // can_be_applied) supplies the offload schedule the copy is parallelized with.
    structured_control_flow::Map* coop_map = nullptr;
    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
        if (auto* enclosing = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
            coop_map = dynamic_cast<structured_control_flow::Map*>(enclosing);
            break;
        }
    }

    // A leading barrier prevents a re-staged (per-thread) tile from being
    // overwritten while the previous coverage iteration's reads are outstanding.
    if (leading_barrier) {
        auto& pre_block = builder.add_block_before(parent, loop_, loop_.debug_info());
        builder.add_library_node<data_flow::BarrierLocalNode>(pre_block, DebugInfo());
    }

    auto c_name = builder.find_new_name("__daisy_ls_coop_" + container_);
    builder.add_container(c_name, types::Scalar(types::PrimitiveType::UInt64));
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

    auto& block = builder.add_block(copy_map.root());
    auto& src = builder.add_access(block, container_);
    auto& dst = builder.add_access(block, local_name_);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Buffer slot offset (per-thread row) plus the flat tile index c.
    data_flow::Subset dst_subset = {symbolic::add(buffer.slot_offset(slot_indices), c)};
    builder.add_computational_memlet(block, src, tasklet, "_in", tile_info_.original_subset(decomp), pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", dst, dst_subset, buffer_type);

    // Barrier so every thread's load is visible before the tile is consumed.
    auto& barrier_block = builder.add_block_before(parent, loop_, loop_.debug_info());
    builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block, DebugInfo());
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
                memlet.set_subset({buffer.linearize(slot_indices, tile_info_.local_index(acc->subset))});
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

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);

    j["subgraph"]["1"] = nlohmann::json::object();
    j["subgraph"]["1"]["element_id"] = access_node_.element_id();
    j["subgraph"]["1"]["type"] = "access_node";
}

} // namespace transformations
} // namespace sdfg
