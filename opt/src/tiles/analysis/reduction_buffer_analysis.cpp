#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"

#include <algorithm>
#include <limits>
#include <set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/utils.h"

/**
 * Reduction footprints describe distinct output addresses, not the number of
 * reduction iterations or the padded span between the first and last address.
 * A scalar accumulator starts with one slot. Each independent inner output axis
 * contributes a positive source stride and trip count; its count multiplies the
 * number of slots, while its initial value shifts the footprint origin.
 *
 * compute() first reconstructs this logical layout, then allocation_cost() assigns
 * private/shared storage according to the schedule and enclosing reduction owners.
 * Exact layouts can be materialized. Proven upper bounds expose costs only, and
 * unsupported expressions leave costs absent rather than claiming zero allocation.
 *
 * After materialization, original_index preserves the logical address expression;
 * the inferred layout is checked against actual declarations and compact memlets.
 * Proposed transformations are analyzed on detached graphs with fresh analyses so
 * current cached results never masquerade as the footprint of a changed loop nest.
 */
namespace sdfg::tiles {

namespace {

/// Find a uniform original index across computational accumulator memlets; reject aliases or inconsistent accesses.
/// Reads and writes must agree, including across nested control flow. This deliberately
/// rejects unions of different access patterns instead of overestimating their span.
class AccumulatorIndexCollector : public analysis::BaseUserVisitor {
    std::string container_;

    void record(const std::string& container, const data_flow::Memlet& edge) {
        if (container != container_) return;
        if (edge.type() != data_flow::MemletType::Computational) {
            throw InvalidSDFGException("accumulator aliases cannot be packed");
        }
        if (edge.subset().size() > 1) {
            throw InvalidSDFGException("unsupported multidimensional accumulator access");
        }
        symbolic::Expression candidate = symbolic::zero();
        if (!edge.subset().empty()) candidate = edge.subset().front();
        if (!index.is_null() && !symbolic::eq(index, candidate)) {
            throw InvalidSDFGException("accumulator is accessed with inconsistent indices in the reduce body");
        }
        index = candidate;
    }

public:
    explicit AccumulatorIndexCollector(const std::string& container) : container_(container) {}
    symbolic::Expression index = SymEngine::null;

    void
    use_as_src_node(const std::string& container, const data_flow::AccessNode&, const data_flow::Memlet& edge, const structured_control_flow::Block&)
        override {
        record(container, edge);
    }
    void
    use_as_dst_node(const std::string& container, const data_flow::AccessNode&, const data_flow::Memlet& edge, const structured_control_flow::Block&)
        override {
        record(container, edge);
    }
    void use_as_symbol_read(
        const std::string&,
        const structured_control_flow::ControlFlowNode*,
        const Element*,
        SymbolReadLocation,
        int,
        symbolic::Expression
    ) override {}
    void use_as_symbol_write(
        const symbolic::Symbol&, const structured_control_flow::ControlFlowNode*, const Element*, SymbolWriteLocation
    ) override {}
    void use_as_return_src(const std::string&, const structured_control_flow::Return&) override {}
};

bool positive_integer(const symbolic::Expression& expression) {
    return !expression.is_null() && SymEngine::is_a<SymEngine::Integer>(*expression) &&
           symbolic::is_positive(expression, {}, {});
}

int64_t checked_product(int64_t left, int64_t right) {
    if (left < 0 || right < 0 || (right != 0 && left > std::numeric_limits<int64_t>::max() / right)) {
        throw InvalidSDFGException("reduction buffer byte count overflows");
    }
    return left * right;
}

bool gpu_loop(const structured_control_flow::StructuredLoop& loop) {
    return gpu::is_gpu_schedule(loop.schedule_type()) &&
           loop.schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader &&
           loop.schedule_type().properties().contains("target_level");
}

bool owns_container(const structured_control_flow::Reduce& reduction, const std::string& container) {
    return std::any_of(reduction.reductions().begin(), reduction.reductions().end(), [&](const auto& entry) {
        return entry.container == container;
    });
}

/// Packed accesses require the same address mapping, not merely the same allocation size.
bool same_layout(const gpu::ReductionLayout& left, const gpu::ReductionLayout& right) {
    if (left.extent != right.extent || !symbolic::eq(left.base, right.base) ||
        left.dimensions.size() != right.dimensions.size())
        return false;
    for (size_t axis = 0; axis < left.dimensions.size(); ++axis) {
        if (left.dimensions[axis].stride != right.dimensions[axis].stride ||
            left.dimensions[axis].count != right.dimensions[axis].count)
            return false;
    }
    return true;
}

/// Charge shared storage to the outermost cooperating owner and suppress redundant private partials.
/// Requires an inferred layout and element width; rejects unsupported strategies or block dimensions.
/// For E output slots of B bytes, the default private contribution is E*B. A Shared
/// owner contributes Tx*Ty*Tz*E*B once; nested Shared reductions of the same accumulator
/// reference it with zero allocation contribution. Cooperative descendants already
/// provide partials, so an enclosing Shared reduction needs no redundant private array.
void allocation_cost(
    ReductionBufferInfo& result,
    structured_control_flow::Reduce& reduction,
    const std::string& container,
    analysis::LoopAnalysis& loops
) {
    auto footprint_bytes = checked_product(result.layout->extent, *result.element_bytes);
    result.private_bytes = footprint_bytes;
    result.shared_bytes = 0;
    if (!gpu_loop(reduction)) return;
    const auto strategy = gpu::ScheduleType_GPU_Offload::partial_storage(reduction.schedule_type());
    const auto level = gpu::ScheduleType_GPU_Offload::target_level(reduction.schedule_type());
    auto placed = gpu::ScheduleType_GPU_Offload::partial_container(reduction.schedule_type());
    if (!placed.empty() && (strategy != gpu::ReduceStrategy::Shared || reduction.reductions().size() != 1)) {
        throw InvalidSDFGException("partial_container requires a single Shared accumulator");
    }
    if (strategy == gpu::ReduceStrategy::Global && level == gpu::TargetLevel::WARP) {
        throw InvalidSDFGException("Global reduction strategy is not supported at WARP scope");
    }
    if (strategy == gpu::ReduceStrategy::Register && level != gpu::TargetLevel::WARP) {
        throw InvalidSDFGException("Register reduction strategy requires WARP scope");
    }
    if (strategy != gpu::ReduceStrategy::Shared) return;
    if (!gpu::is_block_level(level)) {
        throw InvalidSDFGException("Shared reduction strategy requires block scope");
    }
    auto* owner = &reduction;
    for (auto* ancestor : loops.ancestors(&reduction)) {
        auto* enclosing = dyn_cast<structured_control_flow::Reduce*>(ancestor);
        if (!enclosing || !gpu_loop(*enclosing) || !owns_container(*enclosing, container)) continue;
        if (!gpu::is_block_level(gpu::ScheduleType_GPU_Offload::target_level(enclosing->schedule_type()))) {
            continue;
        }
        if (gpu::ScheduleType_GPU_Offload::partial_storage(enclosing->schedule_type()) != gpu::ReduceStrategy::Shared) {
            throw InvalidSDFGException("Shared reduction cannot reuse a Global block owner's partial buffer");
        }
        if (loops.ancestors(enclosing).size() < loops.ancestors(owner).size()) owner = enclosing;
    }
    result.shared_owner = owner->element_id();
    bool nested_cooperative = false;
    for (auto* descendant : loops.descendants(&reduction)) {
        auto* nested = dyn_cast<structured_control_flow::Reduce*>(descendant);
        if (!nested || !gpu_loop(*nested) || !owns_container(*nested, container)) continue;
        auto nested_level = gpu::ScheduleType_GPU_Offload::target_level(nested->schedule_type());
        nested_cooperative |= gpu::is_block_level(nested_level) || gpu::is_warp_level(nested_level);
    }
    if (nested_cooperative || owner != &reduction) result.private_bytes = 0;
    if (owner != &reduction) return;
    // Repeated uses of one block axis must agree on its width and count only once.
    // Include this reduction's ancestors and descendants, not unrelated sibling scopes.
    std::map<gpu::TargetLevel, int64_t> dimensions;
    auto collect = [&](structured_control_flow::StructuredLoop& loop) {
        if (!gpu_loop(loop)) return;
        auto axis = gpu::ScheduleType_GPU_Offload::target_level(loop.schedule_type());
        if (!gpu::is_block_level(axis)) {
            return;
        }
        auto count = gpu::ScheduleType_GPU_Offload::parallel_size(loop.schedule_type())->as_int();
        if (count <= 0 || (dimensions.contains(axis) && dimensions.at(axis) != count)) {
            throw InvalidSDFGException("inconsistent block dimensions for reduction buffer");
        }
        dimensions[axis] = count;
    };
    collect(reduction);
    for (auto* node : loops.ancestors(&reduction)) {
        if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(node)) collect(*loop);
    }
    for (auto* node : loops.descendants(&reduction)) {
        if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(node)) collect(*loop);
    }
    int64_t threads = 1;
    for (const auto& [axis, count] : dimensions) threads = checked_product(threads, count);
    result.shared_bytes = checked_product(threads, footprint_bytes);
    auto block_x =
        symbolic::integer(dimensions.contains(gpu::TargetLevel::X_BLOCK) ? dimensions.at(gpu::TargetLevel::X_BLOCK) : 1);
    auto block_y =
        symbolic::integer(dimensions.contains(gpu::TargetLevel::Y_BLOCK) ? dimensions.at(gpu::TargetLevel::Y_BLOCK) : 1);
    result.linear_thread_index = symbolic::add(
        symbolic::threadIdx_x(),
        symbolic::mul(block_x, symbolic::add(symbolic::threadIdx_y(), symbolic::mul(block_y, symbolic::threadIdx_z())))
    );
}

} // namespace

ReductionBufferAnalysis::ReductionBufferAnalysis(StructuredSDFG& sdfg) : analysis::Analysis(sdfg) {}

std::string ReductionBufferAnalysis::name() const { return "ReductionBufferAnalysis"; }

void ReductionBufferAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    analysis_manager_ = &analysis_manager;
    options_ = &analysis_manager.options();
    buffers_.clear();
}

const ReductionBufferInfo& ReductionBufferAnalysis::
    buffer(structured_control_flow::Reduce& reduction, const std::string& container) const {
    auto key = std::make_pair(&reduction, container);
    auto found = buffers_.find(key);
    if (found == buffers_.end()) {
        found = buffers_.emplace(key, compute(reduction, container)).first;
    }
    return found->second;
}

const ReductionBufferInfo& ReductionBufferAnalysis::
    require(structured_control_flow::Reduce& reduction, const std::string& container) const {
    const auto& result = buffer(reduction, container);
    if (result.status != ReductionBufferStatus::Exact) {
        throw InvalidSDFGException("ReductionBufferAnalysis: accumulator '" + container + "': " + result.diagnostic);
    }
    return result;
}

ReductionBufferInfo ReductionBufferAnalysis::
    compute(structured_control_flow::Reduce& reduction, const std::string& container) const {
    try {
        if (std::none_of(reduction.reductions().begin(), reduction.reductions().end(), [&](const auto& entry) {
                return entry.container == container;
            })) {
            throw InvalidSDFGException("container is not an accumulator of this reduction");
        }
        // After packing, the saved index is authoritative because body memlets address partials.
        AccumulatorIndexCollector collector(container);
        bool materialized = false;
        for (const auto& entry : reduction.reductions()) {
            if (entry.container == container && !entry.original_index.is_null()) {
                collector.index = entry.original_index;
                materialized = true;
            }
        }
        if (!materialized) {
            collector.visit(reduction.root());
        }
        if (collector.index.is_null()) {
            throw InvalidSDFGException("accumulator is not accessed in the reduce body");
        }
        if (symbolic::uses(collector.index, reduction.indvar())) {
            throw InvalidSDFGException("accumulator index depends on the reduction variable (scatter)");
        }
        auto& loop_analysis = analysis_manager_->get<analysis::LoopAnalysis>();
        std::vector<structured_control_flow::StructuredLoop*> inner_loops;
        for (auto* descendant : loop_analysis.descendants(&reduction)) {
            if (auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                inner_loops.push_back(inner);
            }
        }
        // Invariant: base retains only the unresolved axes. For an affine axis i,
        // base = residual + coefficient*i; record (coefficient*step, trip_count),
        // then substitute i's init. Inner-first substitution also handles tiled
        // initial values that depend on an enclosing induction variable. Reversing
        // this order can miss the outer axis introduced by that substitution.
        std::sort(inner_loops.begin(), inner_loops.end(), [&](auto* left, auto* right) {
            return loop_analysis.ancestors(left).size() > loop_analysis.ancestors(right).size();
        });
        auto base = collector.index;
        std::vector<gpu::ReductionLayout::Dimension> dimensions;
        bool exact = true;
        for (auto* inner : inner_loops) {
            if (!symbolic::uses(base, inner->indvar())) {
                continue;
            }
            auto origin = SymEngine::subs(base, {{inner->indvar(), symbolic::zero()}});
            auto coefficient =
                symbolic::expand(symbolic::sub(SymEngine::subs(base, {{inner->indvar(), symbolic::one()}}), origin));
            auto count = inner->num_iterations();
            if (!count.is_null() && !SymEngine::is_a<SymEngine::Integer>(*count)) {
                auto& assumptions = analysis_manager_->get<analysis::AssumptionsAnalysis>();
                symbolic::BoundAnalysis bounds(assumptions.parameters(), assumptions.get(*inner), true);
                auto interval = bounds.bound(count);
                // Equal endpoints prove a fixed trip count even for guarded tiles.
                // An upper endpoint alone bounds allocation but cannot authorize
                // writeback to every slot, since some outputs may not be visited.
                if (interval.has_lower() && interval.has_upper() && symbolic::eq(interval.lower, interval.upper)) {
                    count = interval.lower;
                } else if (interval.has_upper() && positive_integer(interval.upper)) {
                    count = interval.upper;
                    exact = false;
                }
            }
            auto stride = inner->stride();
            if (!positive_integer(count) || !positive_integer(stride) || !positive_integer(coefficient) ||
                !symbolic::
                    eq(symbolic::expand(base),
                       symbolic::expand(symbolic::add(origin, symbolic::mul(coefficient, inner->indvar()))))) {
                throw InvalidSDFGException(
                    "requires a constant positive affine inner-loop footprint for '" + inner->indvar()->get_name() + "'"
                );
            }
            auto coefficient_value = SymEngine::rcp_static_cast<const SymEngine::Integer>(coefficient)->as_int();
            auto stride_value = stride->as_int();
            if (coefficient_value > std::numeric_limits<int64_t>::max() / stride_value) {
                throw InvalidSDFGException("accumulator index stride overflows");
            }
            dimensions.push_back(
                {coefficient_value * stride_value, SymEngine::rcp_static_cast<const SymEngine::Integer>(count)->as_int()
                }
            );
            base = SymEngine::subs(base, {{inner->indvar(), inner->init()}});
        }
        for (auto* inner : inner_loops) {
            if (symbolic::uses(base, inner->indvar())) {
                throw InvalidSDFGException(
                    "footprint origin still depends on inner loop '" + inner->indvar()->get_name() + "'"
                );
            }
        }
        if (symbolic::uses(base, reduction.indvar())) {
            throw InvalidSDFGException("footprint origin depends on the reduction variable");
        }
        const types::IType* type = &sdfg_.type(container);
        if (auto* pointer = dynamic_cast<const types::Pointer*>(type); pointer && !pointer->has_pointee_type()) {
            type = analysis_manager_->get<analysis::TypeAnalysis>().get_outer_type(container);
        }
        if (!type) {
            throw InvalidSDFGException("unresolved accumulator element type");
        }
        auto* scalar = dynamic_cast<const types::Scalar*>(&types::peel_to_innermost_element(*type));
        if (!scalar || scalar->primitive_type() == types::PrimitiveType::Void) {
            throw InvalidSDFGException("accumulator must have a scalar element type");
        }
        ReductionBufferInfo result;
        result.multi_output = !dimensions.empty();
        result.layout.emplace(base, std::move(dimensions));
        result.accumulator_index = collector.index;
        result.primitive = scalar->primitive_type();
        result.element_bytes = (types::bit_width(*result.primitive) + 7) / 8;
        allocation_cost(result, reduction, container, loop_analysis);
        if (!exact) {
            if (materialized) throw InvalidSDFGException("materialized reduction requires an exact footprint");
            result.status = ReductionBufferStatus::ConservativeBound;
            result.diagnostic = "only an upper bound on the reduction footprint is known";
            result.layout.reset();
            return result;
        }
        result.materialized = materialized;
        // The original index, not a previous byte count or packed memlet, defines
        // the footprint after materialization. Check declarations and address maps
        // against fresh inference so changes with equal byte costs are not hidden.
        const auto& properties = reduction.schedule_type().properties();
        auto private_ref = properties.find("reduction_private." + container);
        auto shared_ref = properties.find("reduction_shared." + container);
        if (private_ref != properties.end()) {
            result.private_buffer = private_ref->second;
        }
        if (shared_ref != properties.end()) {
            result.shared_buffer = shared_ref->second;
        }
        if (materialized) {
            auto validate_buffer = [&](const std::string& name, int64_t bytes, bool shared) {
                if (bytes == 0) return;
                if (name.empty() || !sdfg_.exists(name))
                    throw InvalidSDFGException("missing materialized partial buffer");
                auto* array = dynamic_cast<const types::Array*>(&sdfg_.type(name));
                if (!array || !symbolic::eq(array->num_elements(), symbolic::integer(bytes / *result.element_bytes)) ||
                    array->element_type() != types::Scalar(*result.primitive) ||
                    (shared ? !array->storage_type().is_nv_shared() : !array->storage_type().is_cpu_stack())) {
                    throw InvalidSDFGException("incompatible materialized partial buffer '" + name + "'");
                }
            };
            validate_buffer(result.private_buffer, *result.private_bytes, false);
            validate_buffer(result.shared_buffer, *result.shared_bytes, true);
            if (!*result.private_bytes && !result.private_buffer.empty()) {
                throw InvalidSDFGException("unexpected private buffer for shared accumulator");
            }
            if (!result.shared_owner && !result.shared_buffer.empty()) {
                throw InvalidSDFGException("unexpected shared buffer for private accumulator");
            }
            if (result.shared_owner && *result.shared_owner != reduction.element_id()) {
                builder::StructuredSDFGBuilder builder(sdfg_);
                auto* owner = dyn_cast<structured_control_flow::Reduce*>(builder.find_element_by_id(*result.shared_owner
                ));
                if (!owner) {
                    throw InvalidSDFGException("missing shared buffer owner");
                }
                const auto& owner_info = require(*owner, container);
                if (result.shared_buffer != owner_info.shared_buffer) {
                    throw InvalidSDFGException("inconsistent shared buffer owner reference");
                }
            }
            auto validate_accesses = [&](const std::string& name, symbolic::Expression expected) {
                if (name.empty()) {
                    return;
                }
                AccumulatorIndexCollector accesses(name);
                accesses.visit(reduction.root());
                if (!accesses.index.is_null() && !symbolic::eq(accesses.index, expected)) {
                    throw InvalidSDFGException("materialized partial memlets do not match the inferred layout");
                }
            };
            auto packed_index = result.layout->pack(result.accumulator_index);
            validate_accesses(result.private_buffer, packed_index);
            if (*result.shared_bytes) {
                validate_accesses(
                    result.shared_buffer,
                    symbolic::
                        add(packed_index,
                            symbolic::mul(result.linear_thread_index, symbolic::integer(result.layout->extent)))
                );
            }
            AccumulatorIndexCollector original_accesses(container);
            original_accesses.visit(reduction.root());
            if (!original_accesses.index.is_null()) {
                throw InvalidSDFGException("materialized reduction still accesses the original accumulator");
            }
        }
        result.status = ReductionBufferStatus::Exact;
        return result;
    } catch (const InvalidSDFGException& error) {
        ReductionBufferInfo result;
        result.diagnostic = error.what();
        return result;
    }
}

ReductionBufferEstimate ReductionBufferAnalysis::estimate(const ReductionInterchangeProposal& proposal) const {
    // Preserve node IDs on the clone so results can be matched to live reductions.
    // This is a geometry preview, not a dependence-legality check for interchange.
    auto snapshot = sdfg_.clone();
    builder::StructuredSDFGBuilder builder(*snapshot);
    auto* outer = dyn_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(proposal.outer_id));
    auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(proposal.inner_id));
    if (!outer || !inner || outer->root().size() != 1 || &outer->root().at(0) != inner) {
        throw InvalidSDFGException("ReductionBufferAnalysis: proposal requires directly nested loops");
    }
    auto& parent = static_cast<structured_control_flow::Sequence&>(*outer->get_parent());
    auto position = parent.index(*outer);
    builder.move_children(inner->root(), outer->root());
    builder.move_child(outer->root(), 0, parent, position + 1);
    builder.move_child(parent, position, inner->root());
    builder.update_loop(
        *inner, inner->indvar(), proposal.new_outer.condition, proposal.new_outer.init, proposal.new_outer.update
    );
    builder.update_loop(
        *outer, outer->indvar(), proposal.new_inner.condition, proposal.new_inner.init, proposal.new_inner.update
    );
    auto result = estimate(*snapshot);
    for (auto& [key, info] : result) {
        if (info.materialized && (key.first == proposal.outer_id || key.first == proposal.inner_id)) {
            info = ReductionBufferInfo{};
            info.diagnostic = "interchange would move a materialized reduction";
        }
    }
    return result;
}

ReductionBufferEstimate ReductionBufferAnalysis::estimate(const ReductionScheduleProposal& proposal) const {
    auto snapshot = sdfg_.clone();
    builder::StructuredSDFGBuilder builder(*snapshot);
    auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(proposal.loop_id));
    if (!loop) {
        throw InvalidSDFGException("reduction schedule proposal requires a structured loop");
    }
    builder.update_schedule_type(*loop, proposal.schedule);
    return estimate(*snapshot);
}

bool ReductionBufferAnalysis::supports_schedule(
    structured_control_flow::StructuredLoop& loop, const structured_control_flow::ScheduleType& schedule
) const {
    auto& loops = analysis_manager_->get<analysis::LoopAnalysis>();
    auto* root = &loop;
    for (auto* node : loops.ancestors(&loop)) {
        auto* ancestor = dyn_cast<structured_control_flow::StructuredLoop*>(node);
        if (ancestor && gpu_loop(*ancestor) && loops.ancestors(ancestor).size() < loops.ancestors(root).size()) {
            root = ancestor;
        }
    }
    auto affected = affected_reductions(*root);
    if (affected.empty() && !dyn_cast<structured_control_flow::Reduce*>(&loop)) {
        return true;
    }
    for (const auto& [key, info] : estimate(ReductionScheduleProposal{loop.element_id(), schedule})) {
        if (key.first != loop.element_id() &&
            std::none_of(affected.begin(), affected.end(), [&](const auto* reduction) {
                return reduction->element_id() == key.first;
            })) {
            continue;
        }
        if (info.status != ReductionBufferStatus::Exact) {
            return false;
        }
    }
    return true;
}

// Recompute under the proposed nesting; equal byte counts alone do not preserve a packed layout.
// Fresh analyses see the proposal's bounds and schedules rather than the live
// graph's caches. Already-packed graphs must preserve the address map, primitive
// type, and allocation owner as well as cost; otherwise their existing memlets
// would require unpacking and rematerialization, which this preview does not do.
ReductionBufferEstimate ReductionBufferAnalysis::estimate(StructuredSDFG& proposal) const {
    if (&proposal == &sdfg_) throw InvalidSDFGException("reduction preview requires a detached proposal");
    analysis::AnalysisManager manager(proposal, additional_assumptions_, *options_);
    auto& loops = manager.get<analysis::LoopAnalysis>();
    auto& buffers = manager.get<ReductionBufferAnalysis>();
    ReductionBufferEstimate result;
    for (auto* node : loops.loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction || !gpu_loop(*reduction)) {
            continue;
        }
        for (const auto& entry : reduction->reductions()) {
            auto info = buffers.buffer(*reduction, entry.container);
            if (info.materialized && info.status == ReductionBufferStatus::Exact) {
                builder::StructuredSDFGBuilder original_builder(sdfg_);
                auto* original =
                    dyn_cast<structured_control_flow::Reduce*>(original_builder
                                                                   .find_element_by_id(reduction->element_id()));
                if (!original) {
                    info = ReductionBufferInfo{};
                    info.diagnostic = "proposal introduces a materialized reduction without an original owner";
                    result.emplace(std::make_pair(reduction->element_id(), entry.container), std::move(info));
                    continue;
                }
                const auto& before = require(*original, entry.container);
                if (!same_layout(*before.layout, *info.layout) || before.shared_owner != info.shared_owner ||
                    before.private_bytes != info.private_bytes || before.shared_bytes != info.shared_bytes ||
                    before.primitive != info.primitive) {
                    info = ReductionBufferInfo{};
                    info.diagnostic = "proposal would invalidate materialized reduction layout or ownership";
                }
            }
            result.emplace(std::make_pair(reduction->element_id(), entry.container), std::move(info));
        }
    }
    return result;
}

std::vector<structured_control_flow::Reduce*> ReductionBufferAnalysis::
    affected_reductions(structured_control_flow::StructuredLoop& loop) const {
    auto& loops = analysis_manager_->get<analysis::LoopAnalysis>();
    std::vector<structured_control_flow::Reduce*> result;
    const auto& ancestors = loops.ancestors(&loop);
    for (auto* node : loops.loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction || !gpu_loop(*reduction)) {
            continue;
        }
        const auto& parents = loops.ancestors(reduction);
        if (reduction == &loop || std::find(parents.begin(), parents.end(), &loop) != parents.end() ||
            std::find(ancestors.begin(), ancestors.end(), reduction) != ancestors.end())
            result.push_back(reduction);
    }
    return result;
}

bool ReductionBufferAnalysis::is_partial_buffer(const std::string& container) const {
    for (auto* node : analysis_manager_->get<analysis::LoopAnalysis>().loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction || !gpu_loop(*reduction)) {
            continue;
        }
        for (const auto& [key, value] : reduction->schedule_type().properties()) {
            if ((key.starts_with("reduction_private.") || key.starts_with("reduction_shared.")) && value == container)
                return true;
        }
    }
    return false;
}

ReductionKernelInfo ReductionBufferAnalysis::kernel(const structured_control_flow::StructuredLoop& root) const {
    // Sum allocation contributions, not referenced buffers: nested users contribute
    // zero. Unknown costs poison the total; a bounded private footprint with no
    // shared allocation still contributes an exact zero to this shared-only query.
    ReductionKernelInfo result;
    bool exact = true;
    int64_t bytes = 0;
    std::set<std::string> allocated;
    auto& loops = analysis_manager_->get<analysis::LoopAnalysis>();
    for (auto* node : loops.loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction || !gpu_loop(*reduction)) {
            continue;
        }
        const auto& ancestors = loops.ancestors(reduction);
        if (reduction != &root && std::find(ancestors.begin(), ancestors.end(), &root) == ancestors.end()) {
            continue;
        }
        for (const auto& entry : reduction->reductions()) {
            const auto& info = buffer(*reduction, entry.container);
            if (!info.shared_bytes) {
                result.diagnostic = info.diagnostic;
                return result;
            }
            exact &= info.status == ReductionBufferStatus::Exact || *info.shared_bytes == 0;
            if (!*info.shared_bytes) {
                continue;
            }
            if (!info.shared_buffer.empty() && !allocated.insert(info.shared_buffer).second) {
                result.diagnostic = "shared reduction buffer is allocated by multiple owners";
                return result;
            }
            if (*info.shared_bytes > std::numeric_limits<int64_t>::max() - bytes) {
                result.diagnostic = "kernel reduction shared-byte total overflows";
                return result;
            }
            bytes += *info.shared_bytes;
        }
    }
    result.status = exact ? ReductionBufferStatus::Exact : ReductionBufferStatus::ConservativeBound;
    result.shared_bytes = bytes;
    return result;
}

} // namespace sdfg::tiles
