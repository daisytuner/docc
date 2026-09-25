#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

bool is_offloaded_reduction(const structured_control_flow::Reduce& reduction) {
    return reduction.schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader &&
           reduction.schedule_type().properties().contains("target_level");
}

/// Choose and reserve a partial-buffer name, validating any explicitly placed shared declaration.
std::string choose_buffer_name(
    builder::StructuredSDFGBuilder& builder,
    const structured_control_flow::Reduce& reduction,
    const std::string& container,
    const tiles::ReductionBufferInfo& info,
    std::set<std::string>& names,
    bool shared
) {
    std::string name;
    if (shared) {
        name = gpu::ScheduleType_GPU_Offload::partial_container(reduction.schedule_type());
    }
    if (name.empty()) {
        auto prefix = "__daisy_reduce_" + std::string(shared ? "smem_" : "reg_") + container + "_" +
                      std::to_string(reduction.element_id()) + "_";
        name = builder.find_new_name(prefix);
    }
    if (!names.insert(name).second) {
        throw InvalidSDFGException("reduction partial buffer has multiple owners");
    }
    auto& sdfg = builder.subject();
    if (sdfg.exists(name)) {
        auto* array = dynamic_cast<const types::Array*>(&sdfg.type(name));
        auto bytes = shared ? *info.shared_bytes : *info.private_bytes;
        if (!array || !symbolic::eq(array->num_elements(), symbolic::integer(bytes / *info.element_bytes)) ||
            array->element_type() != types::Scalar(*info.primitive) ||
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

    bool visit(structured_control_flow::Reduce& reduction) override {
        if (is_offloaded_reduction(reduction) &&
            std::any_of(reduction.reductions().begin(), reduction.reductions().end(), [&](const auto& entry) {
                return entry.container == original_;
            })) {
            return true;
        }
        return visitor::ActualStructuredSDFGVisitor::visit(reduction);
    }

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
    std::set<std::string> names;
    for (auto* node : loops.loops()) {
        auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!reduction || !is_offloaded_reduction(*reduction)) {
            continue;
        }
        bool pending = false;
        for (const auto& entry : reduction->reductions()) {
            if (entry.original_index.is_null()) {
                pending = true;
                continue;
            }
            const auto& info = analysis.require(*reduction, entry.container);
            if ((*info.private_bytes && !names.insert(info.private_buffer).second) ||
                (*info.shared_bytes && !names.insert(info.shared_buffer).second)) {
                throw InvalidSDFGException("reduction partial buffer has multiple owners");
            }
        }
        if (pending) {
            reductions.push_back(reduction);
        }
    }
    auto nest_id = [&](structured_control_flow::Reduce* reduction) {
        for (auto* ancestor : loops.ancestors(reduction)) {
            if (loops.ancestors(ancestor).empty()) {
                return ancestor->element_id();
            }
        }
        return reduction->element_id();
    };
    std::sort(reductions.begin(), reductions.end(), [&](auto* left, auto* right) {
        return std::make_pair(nest_id(left), loops.ancestors(left).size()) <
               std::make_pair(nest_id(right), loops.ancestors(right).size());
    });
    std::map<std::pair<size_t, std::string>, std::pair<gpu::ReductionLayout, symbolic::Expression>> shared_layouts;
    std::optional<size_t> current_nest;
    bool applied = false;
    for (auto* reduction : reductions) {
        auto nest = nest_id(reduction);
        if (current_nest != nest) {
            shared_layouts.clear();
            current_nest = nest;
        }
        for (const auto& entry : reduction->reductions()) {
            if (!entry.original_index.is_null()) {
                continue;
            }
            auto info = analysis.require(*reduction, entry.container);
            if (*info.private_bytes) {
                info.private_buffer = choose_buffer_name(builder, *reduction, entry.container, info, names, false);
            }
            if (*info.shared_bytes) {
                info.shared_buffer = choose_buffer_name(builder, *reduction, entry.container, info, names, true);
            }
            const auto* layout = &*info.layout;
            auto linear_thread_index = info.linear_thread_index;
            if (info.shared_owner && *info.shared_owner != reduction->element_id()) {
                auto* owner = dyn_cast<structured_control_flow::Reduce*>(builder.find_element_by_id(*info.shared_owner)
                );
                if (!owner) {
                    throw InvalidSDFGException("missing enclosing shared reduction buffer owner");
                }
                info.shared_buffer = owner->schedule_type().properties().at("reduction_shared." + entry.container);
                const auto& owner_layout = shared_layouts.at({*info.shared_owner, entry.container});
                layout = &owner_layout.first;
                linear_thread_index = owner_layout.second;
            }
            auto declare = [&](const std::string& name, int64_t bytes, bool shared) {
                if (bytes != 0 && !sdfg.exists(name)) {
                    builder.add_container(
                        name,
                        types::Array(
                            shared ? types::StorageType::NV_Shared() : types::StorageType::CPU_Stack(),
                            0,
                            "",
                            types::Scalar(*info.primitive),
                            symbolic::integer(bytes / *info.element_bytes)
                        )
                    );
                }
            };
            declare(info.private_buffer, *info.private_bytes, false);
            declare(info.shared_buffer, *info.shared_bytes, true);
            auto schedule = reduction->schedule_type();
            if (!info.private_buffer.empty()) {
                schedule.set_property("reduction_private." + entry.container, info.private_buffer);
            }
            if (!info.shared_buffer.empty()) {
                schedule.set_property("reduction_shared." + entry.container, info.shared_buffer);
            }
            builder.update_schedule_type(*reduction, schedule);
            reduction->original_index(entry.container, info.accumulator_index);
            bool shared = info.private_buffer.empty();
            auto name = shared ? info.shared_buffer : info.private_buffer;
            symbolic::Expression offset = symbolic::zero();
            if (shared) {
                offset = symbolic::mul(linear_thread_index, symbolic::integer(layout->extent));
            }
            RewriteAccesses rewrite(entry.container, name, sdfg.type(name), *layout, offset);
            reduction->root().accept(rewrite);
            if (*info.shared_bytes && !*info.private_bytes) {
                shared_layouts.emplace(
                    std::make_pair(reduction->element_id(), entry.container),
                    std::make_pair(std::move(*info.layout), info.linear_thread_index)
                );
            }
            applied = true;
        }
    }
    if (applied) {
        analysis_manager.invalidate_all();
    }
    return applied;
}

} // namespace passes
} // namespace sdfg
