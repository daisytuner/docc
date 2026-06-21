#include "sdfg/passes/offloading/device_resident_arg_promotion_pass.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
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

/**
 * @brief A single boundary offloading node together with the host argument and
 *        device-buffer container it connects.
 */
struct OffloadRecord {
    structured_control_flow::Block* block;
    offloading::DataOffloadingNode* node;
    std::string host; ///< host-side container (the `_hst` endpoint), empty if none
    std::string dev; ///< device-buffer container (the `_dev` endpoint)
    bool alloc; ///< node allocates the device buffer
    bool transfer; ///< node performs an H2D/D2H copy (carries a host endpoint)
};

/**
 * @brief Collects every boundary offloading node (with its host/device
 *        endpoints) and every container that participates in a reference or
 *        dereference memlet (which would make device-buffer aliasing unsafe).
 */
class OffloadingCollector : public visitor::ImmutableStructuredSDFGVisitor {
private:
    std::vector<OffloadRecord>& records_;
    std::unordered_set<std::string>& aliased_containers_;

public:
    OffloadingCollector(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        std::vector<OffloadRecord>& records,
        std::unordered_set<std::string>& aliased_containers
    )
        : visitor::ImmutableStructuredSDFGVisitor(sdfg, analysis_manager), records_(records),
          aliased_containers_(aliased_containers) {}

    bool accept(structured_control_flow::Block& node) override {
        auto& dataflow = node.dataflow();
        for (auto* libnode : dataflow.library_nodes()) {
            auto* offload = dynamic_cast<offloading::DataOffloadingNode*>(libnode);
            if (offload == nullptr) {
                continue;
            }
            OffloadRecord record;
            record.block = &node;
            record.node = offload;
            record.alloc = offload->is_alloc();
            record.transfer = offload->has_transfer();
            for (const auto& memlet : dataflow.in_edges(*offload)) {
                const auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
                if (src == nullptr) {
                    continue;
                }
                if (memlet.dst_conn() == "_hst") {
                    record.host = src->data();
                } else if (memlet.dst_conn() == "_dev") {
                    record.dev = src->data();
                }
            }
            for (const auto& memlet : dataflow.out_edges(*offload)) {
                const auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
                if (dst == nullptr) {
                    continue;
                }
                if (memlet.src_conn() == "_dev") {
                    record.dev = dst->data();
                }
            }
            records_.push_back(record);
        }
        // Any container reached through a reference/dereference memlet may be an
        // alias of another buffer; eliding its staging would be unsafe.
        for (const auto& memlet : dataflow.edges()) {
            if (memlet.type() == data_flow::Reference || memlet.type() == data_flow::Dereference_Src ||
                memlet.type() == data_flow::Dereference_Dst) {
                if (const auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
                    aliased_containers_.insert(src->data());
                }
                if (const auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
                    aliased_containers_.insert(dst->data());
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

    // Elide the per-call staging (cudaMalloc + copy + cudaFree) for every
    // device-resident argument. The canonical pattern for a resident argument is
    //   arg --(alloc + H2D)--> dev_buf --kernel--> arg (+ D2H + free)
    // and, because the argument now lives on the device, the staging buffer can
    // simply alias the argument: the allocation becomes a reference assignment
    // `dev_buf = arg` (a pointer assignment) and the H2D/D2H copies and the free
    // become dead and are removed. Kernels keep referring to `dev_buf`, which now
    // points directly at the resident argument's memory.
    std::vector<OffloadRecord> records;
    std::unordered_set<std::string> aliased_containers;
    OffloadingCollector collector(sdfg, analysis_manager, records, aliased_containers);
    collector.visit();

    // Map each device buffer to the resident argument(s) that feed it via a copy.
    std::unordered_map<std::string, std::unordered_set<std::string>> dev_to_args;
    for (const auto& record : records) {
        if (record.transfer && !record.host.empty() && !record.dev.empty() && ptr_args.count(record.host) != 0) {
            dev_to_args[record.dev].insert(record.host);
        }
    }

    // A device buffer is eligible for elision only in the clean 1:1 case: it is
    // fed by exactly one resident argument, it is itself a transient (not an
    // argument), and it is not aliased through any reference/dereference memlet
    // (which could have been introduced by buffer reuse in earlier passes).
    std::unordered_map<std::string, std::string> eligible; // dev buffer -> resident arg
    for (const auto& [dev, args] : dev_to_args) {
        if (args.size() != 1) {
            continue;
        }
        if (ptr_args.count(dev) != 0) {
            continue;
        }
        if (aliased_containers.count(dev) != 0) {
            continue;
        }
        eligible.emplace(dev, *args.begin());
    }

    for (const auto& record : records) {
        auto it = eligible.find(record.dev);
        if (it == eligible.end()) {
            continue;
        }
        const std::string& arg = it->second;
        auto* block = record.block;
        if (record.alloc) {
            // Replace the device-buffer allocation with a reference: dev = arg.
            auto ref_type = sdfg.type(arg).clone();
            builder.clear_code_node_legacy(*block, *record.node);
            auto& src = builder.add_access(*block, arg);
            auto& dst = builder.add_access(*block, record.dev);
            builder.add_reference_memlet(*block, src, dst, {symbolic::zero()}, *ref_type);
        } else {
            // H2D / D2H / free are all redundant once the buffer aliases the arg.
            builder.clear_code_node_legacy(*block, *record.node);
        }
    }

    return true;
}

} // namespace passes
} // namespace sdfg
