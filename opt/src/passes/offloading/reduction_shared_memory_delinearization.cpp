#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"

#include <algorithm>
#include <set>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

/// Pending materialization for one accumulator, including resolved allocation ownership.
struct BufferPlan {
    structured_control_flow::Reduce* reduction;
    std::string container;
    tiles::ReductionBufferInfo info;
};

/// Choose and reserve a partial-buffer name, validating any explicitly placed shared declaration.
/// Owner-specific prefixes distinguish pending buffers before they are added to the SDFG.
std::string choose_buffer_name(
    builder::StructuredSDFGBuilder& builder, const BufferPlan& plan, std::set<std::string>& names, bool shared
) {
    std::string name;
    if (shared) {
        name = gpu::ScheduleType_GPU_Offload::partial_container(plan.reduction->schedule_type());
    }
    if (name.empty()) {
        auto prefix = "__daisy_reduce_" + std::string(shared ? "smem_" : "reg_") + plan.container + "_" +
                      std::to_string(plan.reduction->element_id()) + "_";
        name = builder.find_new_name(prefix);
    }
    if (!names.insert(name).second) {
        throw InvalidSDFGException("reduction partial buffer has multiple owners");
    }
    auto& sdfg = builder.subject();
    if (sdfg.exists(name)) {
        auto* array = dynamic_cast<const types::Array*>(&sdfg.type(name));
        auto bytes = shared ? *plan.info.shared_bytes : *plan.info.private_bytes;
        if (!array || !symbolic::eq(array->num_elements(), symbolic::integer(bytes / *plan.info.element_bytes)) ||
            array->element_type() != types::Scalar(*plan.info.primitive) ||
            (shared ? !array->storage_type().is_nv_shared() : !array->storage_type().is_cpu_stack())) {
            throw InvalidSDFGException("reduction partial declaration does not match inferred layout");
        }
    }
    return name;
}

/// Redirect accumulator accesses to packed slots, with a per-thread offset for shared storage.
class RewriteAccesses : public visitor::ActualStructuredSDFGVisitor {
    const std::string& original_;
    const std::string& buffer_;
    const types::IType& type_;
    const gpu::ReductionLayout& layout_;
    symbolic::Expression offset_;

public:
    RewriteAccesses(
        const std::string& original,
        const std::string& buffer,
        const types::IType& type,
        const gpu::ReductionLayout& layout,
        symbolic::Expression offset
    )
        : original_(original), buffer_(buffer), type_(type), layout_(layout), offset_(offset) {}

    bool visit(structured_control_flow::Block& block) override {
        auto& graph = block.dataflow();
        for (auto* access : graph.data_nodes()) {
            if (access->data() != original_) {
                continue;
            }
            auto rewrite = [&](data_flow::Memlet& edge) {
                symbolic::Expression index = symbolic::zero();
                if (!edge.subset().empty()) {
                    index = edge.subset().front();
                }
                edge.set_subset({symbolic::add(offset_, layout_.pack(index))});
                edge.set_base_type(type_);
            };
            for (auto& edge : graph.in_edges(*access)) {
                rewrite(edge);
            }
            for (auto& edge : graph.out_edges(*access)) {
                rewrite(edge);
            }
            access->data(buffer_);
        }
        return true;
    }
};

} // namespace

std::string ReductionSharedMemoryDelinearization::name() { return "ReductionSharedMemoryDelinearization"; }

bool ReductionSharedMemoryDelinearization::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& loops = analysis_manager.get<analysis::LoopAnalysis>();
    auto& analysis = analysis_manager.get<tiles::ReductionBufferAnalysis>();
    std::vector<structured_control_flow::Reduce*> reductions;
    for (auto* node : loops.loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction ||
            reduction->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader ||
            !reduction->schedule_type().properties().contains("target_level")) {
            continue;
        }
        reductions.push_back(reduction);
    }
    std::vector<BufferPlan> plans;
    std::set<std::string> names;
    // Reserve existing owners before assigning names to new buffers; shared children do not allocate.
    for (auto* reduction : reductions) {
        for (const auto& entry : reduction->reductions()) {
            const auto& info = analysis.require(*reduction, entry.container);
            if (!info.materialized) {
                continue;
            }
            if ((*info.private_bytes && !names.insert(info.private_buffer).second) ||
                (*info.shared_bytes && !names.insert(info.shared_buffer).second)) {
                throw InvalidSDFGException("reduction partial buffer has multiple owners");
            }
        }
    }
    for (auto* reduction : reductions) {
        for (const auto& entry : reduction->reductions()) {
            auto info = analysis.require(*reduction, entry.container);
            if (info.materialized) {
                continue;
            }
            BufferPlan plan{reduction, entry.container, std::move(info)};
            if (*plan.info.private_bytes) {
                plan.info.private_buffer = choose_buffer_name(builder, plan, names, false);
            }
            if (*plan.info.shared_bytes) {
                plan.info.shared_buffer = choose_buffer_name(builder, plan, names, true);
            }
            plans.push_back(std::move(plan));
        }
    }
    if (plans.empty()) {
        return false;
    }
    // Resolve nested shared references before declaring any containers or changing schedules.
    for (auto& plan : plans) {
        if (plan.info.shared_owner && *plan.info.shared_owner != plan.reduction->element_id()) {
            auto owner = std::find_if(plans.begin(), plans.end(), [&](const auto& candidate) {
                return candidate.reduction->element_id() == *plan.info.shared_owner &&
                       candidate.container == plan.container;
            });
            if (owner == plans.end()) {
                throw InvalidSDFGException("missing enclosing shared reduction buffer owner");
            }
            plan.info.shared_buffer = owner->info.shared_buffer;
        }
    }
    // Materialize declarations and references while preserving the original writeback index.
    for (const auto& plan : plans) {
        auto declare = [&](const std::string& name, int64_t bytes, bool shared) {
            if (bytes == 0 || sdfg.exists(name)) {
                return;
            }
            builder.add_container(
                name,
                types::Array(
                    shared ? types::StorageType::NV_Shared() : types::StorageType::CPU_Stack(),
                    0,
                    "",
                    types::Scalar(*plan.info.primitive),
                    symbolic::integer(bytes / *plan.info.element_bytes)
                )
            );
        };
        declare(plan.info.private_buffer, *plan.info.private_bytes, false);
        declare(plan.info.shared_buffer, *plan.info.shared_bytes, true);
        auto schedule = plan.reduction->schedule_type();
        if (!plan.info.private_buffer.empty()) {
            schedule.set_property("reduction_private." + plan.container, plan.info.private_buffer);
        }
        if (!plan.info.shared_buffer.empty()) {
            schedule.set_property("reduction_shared." + plan.container, plan.info.shared_buffer);
        }
        builder.update_schedule_type(*plan.reduction, schedule);
        plan.reduction->original_index(plan.container, plan.info.accumulator_index);
    }
    // Rewrite children first so an enclosing reduction cannot capture their accumulator accesses.
    std::sort(plans.begin(), plans.end(), [&](const auto& left, const auto& right) {
        return loops.ancestors(left.reduction).size() > loops.ancestors(right.reduction).size();
    });
    for (const auto& plan : plans) {
        const auto* info = &plan.info;
        bool shared = info->private_buffer.empty();
        if (shared && info->shared_owner && *info->shared_owner != plan.reduction->element_id()) {
            for (const auto& owner : plans) {
                if (owner.reduction->element_id() == *info->shared_owner && owner.container == plan.container) {
                    info = &owner.info;
                    break;
                }
            }
        }
        auto name = shared ? info->shared_buffer : info->private_buffer;
        symbolic::Expression offset = symbolic::zero();
        if (shared) {
            offset = symbolic::mul(info->linear_thread_index, symbolic::integer(info->layout->extent));
        }
        RewriteAccesses rewrite(plan.container, name, sdfg.type(name), *info->layout, offset);
        plan.reduction->root().accept(rewrite);
    }
    analysis_manager.invalidate_all();
    return true;
}

} // namespace passes
} // namespace sdfg
