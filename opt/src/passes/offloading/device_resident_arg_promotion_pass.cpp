#include "sdfg/passes/offloading/device_resident_arg_promotion_pass.h"

#include <string>
#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

/**
 * @brief Finds whether any pointer-argument access node is touched by a node
 *        other than a boundary offloading node (i.e. host-side compute).
 *
 * `visit()` returns true as soon as such a "disqualifying" use is found.
 */
class HostUseFinder : public visitor::ImmutableStructuredSDFGVisitor {
private:
    const std::unordered_set<std::string>& ptr_args_;

    static bool is_offloading(const data_flow::DataFlowNode& node) {
        return dynamic_cast<const offloading::DataOffloadingNode*>(&node) != nullptr;
    }

public:
    HostUseFinder(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        const std::unordered_set<std::string>& ptr_args
    )
        : visitor::ImmutableStructuredSDFGVisitor(sdfg, analysis_manager), ptr_args_(ptr_args) {}

    bool accept(structured_control_flow::Block& node) override {
        auto& dataflow = node.dataflow();
        for (const auto* access : dataflow.data_nodes()) {
            if (ptr_args_.find(access->data()) == ptr_args_.end()) {
                continue;
            }
            // Every neighbor of a promotable argument's access node must be a
            // boundary offloading node.
            for (const auto& memlet : dataflow.out_edges(*access)) {
                if (!is_offloading(memlet.dst())) {
                    return true;
                }
            }
            for (const auto& memlet : dataflow.in_edges(*access)) {
                if (!is_offloading(memlet.src())) {
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace

DeviceResidentArgPromotionPass::DeviceResidentArgPromotionPass(bool is_rocm) : is_rocm_(is_rocm) {}

bool DeviceResidentArgPromotionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Collect pointer arguments.
    std::unordered_set<std::string> ptr_args;
    for (const auto& name : sdfg.arguments()) {
        if (dynamic_cast<const types::Pointer*>(&sdfg.type(name)) != nullptr) {
            ptr_args.insert(name);
        }
    }
    if (ptr_args.empty()) {
        return false;
    }

    // Whole-program predicate: no pointer argument may be touched by host code.
    HostUseFinder finder(sdfg, analysis_manager, ptr_args);
    if (finder.visit()) {
        return false;
    }

    // Commit: promote all pointer arguments to device-resident storage.
    auto device_storage = is_rocm_ ? types::StorageType("AMD_Generic") : types::StorageType::NV_Generic();
    for (const auto& name : ptr_args) {
        auto type = sdfg.type(name).clone();
        type->storage_type(device_storage);
        builder.change_type(name, *type);
    }

    return true;
}

} // namespace passes
} // namespace sdfg
