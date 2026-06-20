#include "sdfg/passes/offloading/device_resident_promotion_pass.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
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
    bool free; ///< node frees the device buffer
    bool h2d; ///< transfer direction is host-to-device
    bool d2h; ///< transfer direction is device-to-host
};

/**
 * @brief A host-side staging allocation (`stdlib::MallocNode`) or deallocation
 *        (`stdlib::FreeNode`). These allocate/free the CPU staging buffers that a
 *        device -> host -> device bounce copies through; they are *not* offloading
 *        nodes.
 */
struct HostStagingRecord {
    structured_control_flow::Block* block;
    data_flow::LibraryNode* node;
    std::string container; ///< the host buffer the node allocates/frees
    bool is_malloc; ///< true for MallocNode, false for FreeNode
};

/**
 * @brief Collects every boundary offloading node (with its host/device
 *        endpoints), every host-side staging malloc/free, every container that
 *        participates in a reference or dereference memlet (which would make
 *        device-buffer aliasing unsafe), and, per container, how many *non-staging*
 *        nodes (kernels/tasklets) read or write it.
 *
 * "Staging" nodes are boundary offloading nodes and host malloc/free nodes; an
 * edge to/from such a node is bookkeeping, not real computation. A container whose
 * only neighbours are staging nodes is therefore never touched by actual compute.
 */
class OffloadingCollector : public visitor::ImmutableStructuredSDFGVisitor {
private:
    std::vector<OffloadRecord>& records_;
    std::vector<HostStagingRecord>& host_staging_;
    std::unordered_set<std::string>& aliased_containers_;
    std::unordered_map<std::string, int>& data_writers_;
    std::unordered_map<std::string, int>& data_readers_;

    static bool is_staging(const data_flow::DataFlowNode& node) {
        return dynamic_cast<const offloading::DataOffloadingNode*>(&node) != nullptr ||
               dynamic_cast<const stdlib::MallocNode*>(&node) != nullptr ||
               dynamic_cast<const stdlib::FreeNode*>(&node) != nullptr;
    }

public:
    OffloadingCollector(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        std::vector<OffloadRecord>& records,
        std::vector<HostStagingRecord>& host_staging,
        std::unordered_set<std::string>& aliased_containers,
        std::unordered_map<std::string, int>& data_writers,
        std::unordered_map<std::string, int>& data_readers
    )
        : visitor::ImmutableStructuredSDFGVisitor(sdfg, analysis_manager), records_(records),
          host_staging_(host_staging), aliased_containers_(aliased_containers), data_writers_(data_writers),
          data_readers_(data_readers) {}

    bool accept(structured_control_flow::Block& node) override {
        auto& dataflow = node.dataflow();
        for (auto* libnode : dataflow.library_nodes()) {
            if (auto* offload = dynamic_cast<offloading::DataOffloadingNode*>(libnode)) {
                OffloadRecord record;
                record.block = &node;
                record.node = offload;
                record.alloc = offload->is_alloc();
                record.transfer = offload->has_transfer();
                record.free = offload->is_free();
                record.h2d = offload->is_h2d();
                record.d2h = offload->is_d2h();
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
            } else if (dynamic_cast<stdlib::MallocNode*>(libnode) != nullptr) {
                for (const auto& memlet : dataflow.out_edges(*libnode)) {
                    if (const auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
                        host_staging_.push_back({&node, libnode, dst->data(), true});
                    }
                }
            } else if (dynamic_cast<stdlib::FreeNode*>(libnode) != nullptr) {
                for (const auto& memlet : dataflow.in_edges(*libnode)) {
                    if (const auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
                        host_staging_.push_back({&node, libnode, src->data(), false});
                    }
                }
            }
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
        // Count real (non-staging) readers/writers per container: an in-edge from a
        // non-staging node writes the container, an out-edge to a non-staging node
        // reads it.
        for (const auto* access : dataflow.data_nodes()) {
            for (const auto& memlet : dataflow.in_edges(*access)) {
                if (!is_staging(memlet.src())) {
                    data_writers_[access->data()]++;
                }
            }
            for (const auto& memlet : dataflow.out_edges(*access)) {
                if (!is_staging(memlet.dst())) {
                    data_readers_[access->data()]++;
                }
            }
        }
        return false;
    }
};

/**
 * @brief Eliminate device -> host -> device transient bounces.
 *
 * For a host staging buffer `H` that is never touched by host compute and sits on
 * a single round trip `S --D2H--> H --H2D--> T` (S, T distinct device transients),
 * make `T` alias `S` and drop the staging. Returns true if anything was rewritten.
 *
 * Safety relies on a strict, structurally-checked topology:
 *  - `H` has a malloc and a free, and zero real (non-staging) readers/writers.
 *  - exactly one D2H into `H` and one H2D out of `H`.
 *  - `S` is written by exactly one real producer and read by no real consumer
 *    (its only reader is the D2H), and has exactly one D2H and one free.
 *  - `T` is written by no real producer (its only writer is the H2D) and read by
 *    at least one real consumer, and has exactly one alloc, one H2D and one free.
 *  - neither `S` nor `T` is an argument, arg-aliased, or reference/dereference
 *    aliased.
 * Under these conditions the round trip carries `S`'s single value into `T`, so
 * `T = S` is sound, and removing the *early* free of `S` while keeping the *late*
 * free of `T` leaves exactly one deallocation after the last use — no dangling
 * free, no double free.
 */
bool eliminate_transient_bounces(
    builder::StructuredSDFGBuilder& builder,
    StructuredSDFG& sdfg,
    const std::vector<OffloadRecord>& records,
    const std::vector<HostStagingRecord>& host_staging,
    const std::unordered_set<std::string>& aliased_containers,
    const std::unordered_map<std::string, int>& data_writers,
    const std::unordered_map<std::string, int>& data_readers,
    const std::unordered_set<std::string>& ptr_args,
    const std::unordered_set<std::string>& arg_eligible_devs
) {
    auto count = [](const std::unordered_map<std::string, int>& m, const std::string& k) {
        auto it = m.find(k);
        return it == m.end() ? 0 : it->second;
    };

    // Index host staging nodes by container.
    std::unordered_map<std::string, const HostStagingRecord*> host_malloc;
    std::unordered_map<std::string, const HostStagingRecord*> host_free;
    for (const auto& hs : host_staging) {
        (hs.is_malloc ? host_malloc : host_free)[hs.container] = &hs;
    }

    // Index offloading records by host / device endpoint and role.
    std::unordered_map<std::string, std::vector<const OffloadRecord*>> host_d2h, host_h2d;
    std::unordered_map<std::string, std::vector<const OffloadRecord*>> dev_alloc, dev_free, dev_d2h, dev_h2d;
    for (const auto& r : records) {
        if (r.transfer && r.d2h && !r.host.empty() && !r.dev.empty()) {
            host_d2h[r.host].push_back(&r);
            dev_d2h[r.dev].push_back(&r);
        }
        if (r.transfer && r.h2d && !r.host.empty() && !r.dev.empty()) {
            host_h2d[r.host].push_back(&r);
            dev_h2d[r.dev].push_back(&r);
        }
        if (r.alloc && !r.dev.empty()) {
            dev_alloc[r.dev].push_back(&r);
        }
        if (r.free && !r.dev.empty()) {
            dev_free[r.dev].push_back(&r);
        }
    }

    bool changed = false;
    std::unordered_set<std::string> used_devs;
    std::unordered_set<const data_flow::LibraryNode*> cleared;

    auto clear_node = [&](structured_control_flow::Block* block, data_flow::LibraryNode* n) {
        if (n == nullptr || cleared.count(n) != 0) {
            return;
        }
        builder.clear_code_node_legacy(*block, *n);
        cleared.insert(n);
    };

    for (const auto& [H, d2hs] : host_d2h) {
        // H must be a pure host staging buffer, untouched by real compute.
        if (ptr_args.count(H) != 0 || aliased_containers.count(H) != 0) {
            continue;
        }
        if (host_malloc.find(H) == host_malloc.end() || host_free.find(H) == host_free.end()) {
            continue;
        }
        if (count(data_writers, H) != 0 || count(data_readers, H) != 0) {
            continue;
        }
        auto h2d_it = host_h2d.find(H);
        if (h2d_it == host_h2d.end() || d2hs.size() != 1 || h2d_it->second.size() != 1) {
            continue;
        }
        const OffloadRecord* d2h = d2hs[0];
        const OffloadRecord* h2d = h2d_it->second[0];
        const std::string S = d2h->dev; // source device buffer
        const std::string T = h2d->dev; // destination device buffer
        if (S == T || S.empty() || T.empty()) {
            continue;
        }
        if (used_devs.count(S) != 0 || used_devs.count(T) != 0) {
            continue;
        }
        if (ptr_args.count(S) != 0 || ptr_args.count(T) != 0) {
            continue;
        }
        if (arg_eligible_devs.count(S) != 0 || arg_eligible_devs.count(T) != 0) {
            continue;
        }
        if (aliased_containers.count(S) != 0 || aliased_containers.count(T) != 0) {
            continue;
        }
        // Topology: S produced once and read only by the D2H; T written only by the
        // H2D and read by at least one real consumer.
        if (count(data_writers, S) != 1 || count(data_readers, S) != 0) {
            continue;
        }
        if (count(data_writers, T) != 0 || count(data_readers, T) < 1) {
            continue;
        }
        // S must have exactly one D2H and T exactly one H2D (no other bounces).
        if (dev_d2h[S].size() != 1 || dev_h2d[T].size() != 1) {
            continue;
        }
        // Need T's allocation, S's (early) free and T's (late) free to be unique.
        auto alloc_it = dev_alloc.find(T);
        auto free_s_it = dev_free.find(S);
        auto free_t_it = dev_free.find(T);
        if (alloc_it == dev_alloc.end() || alloc_it->second.size() != 1) {
            continue;
        }
        if (free_s_it == dev_free.end() || free_s_it->second.size() != 1) {
            continue;
        }
        if (free_t_it == dev_free.end() || free_t_it->second.size() != 1) {
            continue;
        }
        const OffloadRecord* alloc_t = alloc_it->second[0];
        const OffloadRecord* free_s = free_s_it->second[0];

        // Commit: T's allocation becomes a reference `T = S`.
        auto ref_type = sdfg.type(S).clone();
        clear_node(alloc_t->block, alloc_t->node);
        auto& src = builder.add_access(*alloc_t->block, S);
        auto& dst = builder.add_access(*alloc_t->block, T);
        builder.add_reference_memlet(*alloc_t->block, src, dst, {symbolic::zero()}, *ref_type);

        // Drop both copies, the host staging malloc/free and S's early free. T's
        // free is intentionally kept: it now frees S's allocation exactly once,
        // after the last consumer of T (== S).
        clear_node(d2h->block, d2h->node);
        if (h2d->node != alloc_t->node) {
            clear_node(h2d->block, h2d->node);
        }
        clear_node(free_s->block, free_s->node);
        clear_node(host_malloc[H]->block, host_malloc[H]->node);
        clear_node(host_free[H]->block, host_free[H]->node);

        used_devs.insert(S);
        used_devs.insert(T);
        changed = true;
    }

    return changed;
}

} // namespace

DeviceResidentPromotionPass::DeviceResidentPromotionPass(bool is_rocm) : is_rocm_(is_rocm) {}

bool DeviceResidentPromotionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    arguments_promoted_ = false;
    bool changed = false;

    // Collect pointer arguments.
    std::unordered_set<std::string> ptr_args;
    for (const auto& name : sdfg.arguments()) {
        if (dynamic_cast<const types::Pointer*>(&sdfg.type(name)) != nullptr) {
            ptr_args.insert(name);
        }
    }

    // ---------------------------------------------------------------------
    // Phase 1: promote pointer arguments to device-resident storage and elide
    // their per-call staging. This is the part that determines whether the whole
    // program is reported as device resident.
    // ---------------------------------------------------------------------
    // Whole-program predicate: no pointer argument may be touched by host code.
    HostUseFinder finder(sdfg, analysis_manager, ptr_args);
    bool promote_args = !ptr_args.empty() && !finder.visit();

    std::unordered_set<std::string> arg_eligible_devs;
    if (promote_args) {
        // Commit: promote all pointer arguments to device-resident storage.
        auto device_storage = is_rocm_ ? types::StorageType("AMD_Generic") : types::StorageType::NV_Generic();
        for (const auto& name : ptr_args) {
            auto type = sdfg.type(name).clone();
            type->storage_type(device_storage);
            builder.change_type(name, *type);
        }
        arguments_promoted_ = true;
        changed = true;

        // Elide the per-call staging (cudaMalloc + copy + cudaFree) for every
        // device-resident argument. The canonical pattern for a resident argument is
        //   arg --(alloc + H2D)--> dev_buf --kernel--> arg (+ D2H + free)
        // and, because the argument now lives on the device, the staging buffer can
        // simply alias the argument: the allocation becomes a reference assignment
        // `dev_buf = arg` (a pointer assignment) and the H2D/D2H copies and the free
        // become dead and are removed. Kernels keep referring to `dev_buf`, which now
        // points directly at the resident argument's memory.
        std::vector<OffloadRecord> records;
        std::vector<HostStagingRecord> host_staging;
        std::unordered_set<std::string> aliased_containers;
        std::unordered_map<std::string, int> data_writers;
        std::unordered_map<std::string, int> data_readers;
        OffloadingCollector
            collector(sdfg, analysis_manager, records, host_staging, aliased_containers, data_writers, data_readers);
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
            arg_eligible_devs.insert(dev);
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
    }

    // ---------------------------------------------------------------------
    // Phase 2: eliminate purely-internal transient host bounces. This is an
    // additional optimization that does NOT affect device-residency reporting.
    //
    // The pattern is a device buffer S whose data is copied to the host only to be
    // copied straight back into a different device buffer T:
    //   producer --> S --D2H--> H(host) --H2D--> T --consumers
    // where H is a staging buffer never touched by host compute. Because the round
    // trip moves bytes between two device buffers through host memory, T can simply
    // alias S: the allocation of T becomes `T = S`, the host staging (malloc/free),
    // both copies and the *early* free of S are removed, and the *late* free of T
    // (which now refers to S's memory) becomes the single deallocation.
    //
    // The topology is re-collected here because phase 1 may have mutated the graph.
    {
        std::vector<OffloadRecord> records;
        std::vector<HostStagingRecord> host_staging;
        std::unordered_set<std::string> aliased_containers;
        std::unordered_map<std::string, int> data_writers;
        std::unordered_map<std::string, int> data_readers;
        OffloadingCollector
            collector(sdfg, analysis_manager, records, host_staging, aliased_containers, data_writers, data_readers);
        collector.visit();

        if (eliminate_transient_bounces(
                builder,
                sdfg,
                records,
                host_staging,
                aliased_containers,
                data_writers,
                data_readers,
                ptr_args,
                arg_eligible_devs
            )) {
            changed = true;
        }
    }

    return changed;
}

} // namespace passes
} // namespace sdfg
