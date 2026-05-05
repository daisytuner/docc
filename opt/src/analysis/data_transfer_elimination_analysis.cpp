#include "sdfg/analysis/data_transfer_elimination_analysis.h"


#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg::analysis {

void OffloadHolder::remove_h2d_parts() {
    host_data = nullptr;
    host_access = nullptr;
    updates_on_dev = false;
}

void OffloadHolder::remove_d2h_parts() {
    host_data = nullptr;
    host_access = nullptr;
    updates_on_host = false;
}

OffloadState::OffloadState(DataTransferEliminationCandidateCollector& collector) : collector_(collector) {}

void OffloadState::found_escape(const std::string& container) { kills_containers_.insert(container); }

void OffloadState::found_ptr_write(const std::string& container, const data_flow::Memlet* memlet) {
    kills_containers_.insert(container);
}

void OffloadState::found_ptr_read(const std::string& container, const data_flow::Memlet* memlet) {
    // todo check with generated set if its ever possible for it to contain the transfer and a use
    reads_[container].push_back(memlet);
}

void OffloadState::found_full_barrier(ControlFlowNode& node) {
    generated_.clear(); // all generateds are from before and now wiped
    full_kill_ = true;
}

/**
 * @return { type, killing node }
 */
std::pair<OffloadState::KillingType, OffloadHolder*> OffloadState::find_killing_entry_node(const ExposedOffload&
                                                                                               in_flight) const {
    auto& holder = *in_flight.offload;
    static const types::Scalar void_type(types::Void);
    auto& host_access_type = holder.host_access ? holder.host_access->base_type() : void_type;

    auto* offload_node = holder.offload_node;
    auto* malloc_node = holder.malloc_node;

    if (holder.ends_dev_lifetime || holder.updates_on_host || malloc_node) {
        for (const auto& entry : h2d_nodes_ | std::views::values) {
            auto& entry_holder = *entry;
            auto& entry_host_access_type = entry_holder.host_access ? entry_holder.host_access->base_type() : void_type;
            bool host_ptr_matches = entry_holder.host_data && in_flight.container == entry_holder.host_data->data();

            if (host_ptr_matches) {
                if (offload_node && (holder.updates_on_host || holder.ends_dev_lifetime)) {
                    // D2H -> H2D
                    bool is_elim_candidate = holder.offload_node->is_compatible_with(*entry_holder.offload_node) &&
                                             entry_holder.updates_on_dev;
                    return {is_elim_candidate ? KillingType::DeviceReuse : KillingType::Basic, &entry_holder};
                } else if (malloc_node) {
                    // Malloc -> H2D
                    // mallocs should only be in flight as long as they are untouched
                    bool can_be_removed = entry_holder.offload_node->is_alloc() && entry_holder.offload_node->is_h2d();
                    return {can_be_removed ? KillingType::EmptyHostMalloc : KillingType::Basic, &entry_holder};
                } else {
                    return {KillingType::Basic, &entry_holder};
                }
            } else if (host_access_type == entry_host_access_type) { // aliases
                // any -> any with aliasing types
                // return &entry_node; // TODO left unhandled for now, because then most situations like a matmul could
                // never be eliminated
            }
        }
    } else if (holder.starts_dev_lifetime) {
        for (const auto& entry : generated_ | std::views::values) {
            auto& entry_holder = *entry;

            bool dev_ptr_matches = false;
            if (holder.dev_data && entry_holder.dev_data) {
                dev_ptr_matches = in_flight.container == entry_holder.dev_data->data();
            }

            if (dev_ptr_matches) {
                // D_ALLOC -> D_FREE is the expected case, but kill for any match
                KillingType killType = KillingType::Basic;
                if (holder.offload_node->is_compatible_with(*entry_holder.offload_node) &&
                    entry_holder.ends_dev_lifetime) {
                    if (entry_holder.updates_on_host) {
                        killType = KillingType::RedundantD2H;
                    } else {
                        killType = KillingType::DeviceCleanFree;
                    }
                }

                return {killType, &entry_holder};
            }
        }
    }

    return {KillingType::None, nullptr};
}

void OffloadState::found_malloc(Block& block, stdlib::MallocNode& malloc) {
    auto& dflow = block.dataflow();

    auto out_edges = dflow.out_edges_for_connector(malloc, malloc.output(0));
    if (out_edges.size() != 1) {
        throw std::runtime_error(
            "Unsupported: malloc node " + std::to_string(malloc.element_id()) + " with other than 1 output"
        );
    }
    auto* memlet = out_edges.at(0);
    auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&memlet->dst());
    generated_.emplace(malloc.element_id(), std::make_unique<OffloadHolder>(&malloc, access_node, memlet));
}

void OffloadState::found_offload_node(Block& block, offloading::DataOffloadingNode& offload) {
    auto& dflow = block.dataflow();

    bool src_is_dev = false;
    bool src_is_host = false;
    bool dst_is_dev = false;
    bool dst_is_host = false;

    bool starts_dev_lifetime = false;
    bool ends_dev_lifetime = false;
    bool updates_on_dev = false;
    bool updates_on_host = false;

    auto transfer_direction = offload.transfer_direction();
    auto lifecycle = offload.buffer_lifecycle();
    if (transfer_direction == offloading::DataTransferDirection::D2H) {
        src_is_dev = true;
        dst_is_host = true;
        updates_on_host = true;
        if (lifecycle == offloading::BufferLifecycle::FREE) {
            ends_dev_lifetime = true;
        }
    } else if (transfer_direction == offloading::DataTransferDirection::H2D) {
        src_is_host = true;
        dst_is_dev = true;
        updates_on_dev = true;
        if (lifecycle == offloading::BufferLifecycle::ALLOC) {
            starts_dev_lifetime = true;
        }
    } else if (offloading::is_NONE(transfer_direction)) {
        if (lifecycle == offloading::BufferLifecycle::ALLOC) {
            starts_dev_lifetime = true;
            dst_is_dev = true;
        } else if (lifecycle == offloading::BufferLifecycle::FREE) {
            ends_dev_lifetime = true;
            src_is_dev = true;
        }
    }

    const data_flow::AccessNode* found_dev_access = nullptr;
    const data_flow::AccessNode* found_host_access = nullptr;
    const data_flow::Memlet* found_host_memlet = nullptr;

    for (auto& conn : offload.inputs()) {
        auto* memlet = dflow.in_edge_for_connector(offload, conn);
        auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&memlet->src());

        if (src_is_host) {
            found_host_access = access_node;
            found_host_memlet = memlet;
        }
        if (src_is_dev) {
            found_dev_access = access_node;
        }
    }

    for (auto& conn : offload.outputs()) {
        auto edges = dflow.out_edges_for_connector(offload, conn);
        if (edges.size() > 1) {
            throw std::runtime_error(
                "Unsupported: offload node " + std::to_string(offload.element_id()) +
                " with multiple outputs edges on " + conn
            );
        }
        auto* memlet = edges.at(0);
        auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&memlet->dst());

        if (dst_is_host) {
            found_host_access = access_node;
            found_host_memlet = memlet;
        }
        if (dst_is_dev) {
            found_dev_access = access_node;
        }
    }

    if (ends_dev_lifetime || updates_on_host) {
        generated_.emplace(
            offload.element_id(),
            std::make_unique<OffloadHolder>(
                &offload,
                found_host_access,
                found_host_memlet,
                found_dev_access,
                starts_dev_lifetime,
                ends_dev_lifetime,
                updates_on_dev,
                updates_on_host
            )
        );
    } else if (starts_dev_lifetime || updates_on_dev) {
        add_h2d_entry(OffloadHolder{
            &offload,
            found_host_access,
            found_host_memlet,
            found_dev_access,
            starts_dev_lifetime,
            ends_dev_lifetime,
            updates_on_dev,
            updates_on_host
        });
    }
}

void OffloadState::add_h2d_entry(const OffloadHolder& entry) {
    h2d_nodes_.emplace(entry.offload_node->element_id(), std::make_unique<OffloadHolder>(entry));
    // todo also need to remove generated ones killed by this. But right now, only max 1 per block anyway
}

void OffloadState::apply_kills_and_changes(ExposedType& exposed) const {
    if (full_kill_) {
        exposed.clear();
        return;
    }
    std::list<ExposedOffload> dynamic_inserts;
    for (auto it = exposed.begin(); it != exposed.end();) {
        auto& [id, exposedOffload] = *it;
        auto& holder = *exposedOffload.offload;


        auto container_reads = reads_.find(exposedOffload.container);
        if (container_reads != reads_.end() && !container_reads->second.empty()) {
            if (holder.malloc_node) { // mallocs are just killed on first use
                DEBUG_PRINTLN(
                    "In-flight malloc area of #" << holder.malloc_node->element_id()
                                                 << " is read without being initialized!"
                );
                it = exposed.erase(it);
                continue;
            } else { // track if a live var is read
                ++exposedOffload.read_count;
            }
        }

        if (kills_containers_.contains(exposedOffload.container)) {
            it = exposed.erase(it);
            continue;
        }

        auto [kill_type, killing_entry] = find_killing_entry_node(exposedOffload);
        if (kill_type != KillingType::None) {
            if (kill_type == KillingType::DeviceReuse) {
                collector_.found_transfer_reuse_pair(exposedOffload, *killing_entry);
            } else if (kill_type == KillingType::EmptyHostMalloc) {
                collector_.found_empty_host_malloc(exposedOffload, *killing_entry);
            } else if (kill_type == KillingType::DeviceCleanFree) {
                // we have a on-device-alloc that survived without kills to the on-device-free
                // -> promote this to a host-relevant D2H point, that might be reused

                // replace the current H2D with the "D2H" that would allow it to live on
                // this creates a D2H-like exposedOffload, despite us not knowing the host-var at this point
                auto* host_data = holder.host_data;
                if (host_data) {
                    dynamic_inserts.emplace_back(killing_entry, host_data->data(), 0);
                    it = exposed.erase(it);
                    continue;
                }
            } else if (kill_type == KillingType::RedundantD2H) {
                collector_.found_redundant_d2h_pair(exposedOffload, *killing_entry);
            }
            it = exposed.erase(it);
            continue;
        }
        ++it;
    }

    for (auto& [id, gen] : generated_) {
        auto* holder = gen.get();
        if (holder->updates_on_host || holder->malloc_node) { // block unidentified host-container ones from being
                                                              // exposed. If we could reconstruct, it will be a
                                                              // dynamic_insert
            exposed.insert({id, ExposedOffload{holder, holder->host_data->data(), 0}});
        }
    }
    for (auto& gen : dynamic_inserts) {
        exposed.insert({gen.offload->offload_node->element_id(), gen});
    }
    for (auto& [id, gen] : h2d_nodes_) {
        auto* holder = gen.get();
        if (holder->starts_dev_lifetime) {
            exposed.insert({id, ExposedOffload{holder, holder->dev_data->data(), 0}});
        }
    }
}


void DataTransferEliminationAnalysis::handle_lib_node(Block& block, data_flow::LibraryNode& node) {
    BaseUserVisitor::handle_lib_node(block, node);

    if (auto* offload_node = dynamic_cast<offloading::DataOffloadingNode*>(&node)) {
        get_or_create_state(block).found_offload_node(block, *offload_node);
    } else if (auto* malloc_node = dynamic_cast<stdlib::MallocNode*>(&node)) {
        get_or_create_state(block).found_malloc(block, *malloc_node);
    }
}

void DataTransferEliminationAnalysis::handle_structured_loop_before_body(StructuredLoop& loop) {
    BaseUserVisitor::handle_structured_loop_before_body(loop);

    // auto* map = dynamic_cast<sdfg::structured_control_flow::Map*>(&loop);

    // if (map && map->schedule_type().category() == ScheduleTypeCategory::Offloader) {
    //     get_or_create_state(loop).found_offloaded_kernel(*map);
    // }
}

void DataTransferEliminationAnalysis::
    on_escape(const std::string& container, const ControlFlowNode* node, const Element* user) {
    if (dynamic_cast<const Block*>(node)) {
        if (auto* memlet = dynamic_cast<const data_flow::Memlet*>(user)) {
            if (auto* offload = dynamic_cast<const offloading::DataOffloadingNode*>(&memlet->dst())) {
                // accesses of offloading nodes are handled more intelligently in found_offload_node, so ignore them
                // here
                return;
            }
        }
    }
    get_or_create_state(*node).found_escape(container);
}

void DataTransferEliminationAnalysis::
    on_read_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) {
    if (!dynamic_cast<const offloading::DataOffloadingNode*>(&user->dst())) {
        get_or_create_state(*node).found_ptr_read(container, user);
    }
}

void DataTransferEliminationAnalysis::
    on_write_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) {
    if (!dynamic_cast<const offloading::DataOffloadingNode*>(&user->dst())) {
        get_or_create_state(*node).found_ptr_write(container, user);
    }
}

std::unique_ptr<OffloadState> DataTransferEliminationAnalysis::
    create_initial_state(const structured_control_flow::ControlFlowNode& node) {
    return std::make_unique<OffloadState>(*this);
}

void DataTransferEliminationAnalysis::run() {
    dispatch(sdfg_.root());

    run_forward(sdfg_.root());
}

} // namespace sdfg::analysis
