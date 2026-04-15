#include "sdfg/analysis/data_transfer_elimination_analysis.h"


#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg::analysis {

void OffloadHolder::remove_host_side() {
    host_data = nullptr;
    host_access = nullptr;
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
    auto& host_access_type = holder.host_access->base_type();

    auto* offload_node = holder.offload_node;
    auto* malloc_node = holder.malloc_node;

    for (const auto& entry_node : h2d_nodes_ | std::views::values) {
        auto& entry_holder = *entry_node;
        if (entry_holder.host_data->data() == holder.host_data->data()) {
            if (offload_node) {
                bool is_elim_candidate = holder.offload_node->redundant_with(*entry_holder.offload_node);
                return {is_elim_candidate ? KillingType::DeviceReuse : KillingType::Basic, &entry_holder};
            } else if (malloc_node) { // mallocs should only be in flight as long as they are untouched
                bool can_be_removed = entry_holder.offload_node->is_alloc() && entry_holder.offload_node->is_h2d();
                return {can_be_removed ? KillingType::EmptyHostMalloc : KillingType::Basic, &entry_holder};
            }
        } else if (host_access_type == entry_holder.host_access->base_type()) { // aliases
            // return &entry_node; // TODO left unhandled for now, because then most situations like a matmul could
            // never be eliminated
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

    if (is_D2H(offload.transfer_direction())) {
        src_is_dev = true;
        dst_is_host = true;
    } else if (is_H2D(offload.transfer_direction())) {
        src_is_host = true;
        dst_is_dev = true;
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

    if (found_host_access && found_dev_access) {
        if (dst_is_host) {
            generated_.emplace(
                offload.element_id(),
                std::make_unique<OffloadHolder>(&offload, found_host_access, found_host_memlet, found_dev_access)
            );
        } else {
            add_h2d_entry(OffloadHolder{&offload, found_host_access, found_host_memlet, found_dev_access});
        }
    }
}

void OffloadState::add_h2d_entry(const OffloadHolder& entry) {
    h2d_nodes_.emplace(entry.offload_node->element_id(), std::make_unique<OffloadHolder>(entry));
    // todo also need to remove generated ones killed by this. But right now
}

void OffloadState::apply_kills_and_changes(ExposedType& exposed) const {
    if (full_kill_) {
        exposed.clear();
        return;
    }
    for (auto it = exposed.begin(); it != exposed.end();) {
        auto& [id, exposedOffload] = *it;
        auto& holder = *exposedOffload.offload;

        auto* host = exposedOffload.offload->host_data;
        auto& host_container = host->data();

        auto host_reads = reads_.find(host_container);
        if (host_reads != reads_.end() && host_reads->second.size() > 0) {
            if (holder.offload_node) {
                ++exposedOffload.read_count;
            } else if (holder.malloc_node) { // mallocs are just killed on first
                DEBUG_PRINTLN(
                    "In-flight malloc area of #" << holder.malloc_node->element_id()
                                                 << " is read without being initialized!"
                );
                it = exposed.erase(it);
                continue;
            }
        }

        if (kills_containers_.contains(host_container)) {
            it = exposed.erase(it);
            continue;
        }

        auto [kill_type, killing_entry] = find_killing_entry_node(exposedOffload);
        if (kill_type != KillingType::None) {
            if (kill_type == KillingType::DeviceReuse) {
                collector_.found_transfer_reuse_pair(exposedOffload, *killing_entry);
            } else if (kill_type == KillingType::EmptyHostMalloc) {
                collector_.found_empty_host_malloc(exposedOffload, *killing_entry);
            }
            it = exposed.erase(it);
            continue;
        }
        ++it;
    }

    for (auto& [id, gen] : generated_) {
        exposed.insert({id, ExposedOffload{gen.get(), 0}});
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
