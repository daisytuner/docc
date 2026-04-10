#include "sdfg/analysis/data_transfer_elimination_analysis.h"


#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg::analysis {

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
 * @return { candidate_for_elimination, killing node }
 */
std::pair<bool, const OffloadHolder*> OffloadState::find_killing_entry_node(const OffloadHolder& exit_node) const {
    auto& host_access_type = exit_node.host_access->base_type();
    for (const auto& entry_node : kernel_entry_nodes_ | std::views::values) {
        if (entry_node.host_data->data() == exit_node.host_data->data()) {
            bool is_elim_candidate = exit_node.node->redundant_with(*entry_node.node);
            return {is_elim_candidate, &entry_node};
        } else if (host_access_type == entry_node.host_access->base_type()) { // aliases
            // return &entry_node; // TODO left unhandled for now, because then most situations like a matmul could
            // never be eliminated
        }
    }

    return {false, nullptr};
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
    kernel_entry_nodes_.emplace(entry.node->element_id(), entry);
    // todo also need to remove generated ones killed by this. But right now
}

ExposedOffload OffloadState::expose(OffloadHolder& holder) { return ExposedOffload{&holder, 0}; }

void OffloadState::apply_kills_and_changes(ExposedType& exposed) const {
    if (full_kill_) {
        exposed.clear();
        return;
    }
    for (auto it = exposed.begin(); it != exposed.end();) {
        auto& [id, exposedOffload] = *it;
        auto* host = exposedOffload.offload->host_data;
        auto& host_container = host->data();

        auto host_reads = reads_.find(host_container);
        if (host_reads != reads_.end() && host_reads->second.size() > 0) {
            ++exposedOffload.read_count;
        }

        if (host && kills_containers_.contains(host_container)) {
            it = exposed.erase(it);
            continue;
        }

        auto [is_elim_candidate, killing_entry] = find_killing_entry_node(*exposedOffload.offload);
        if (killing_entry) {
            if (is_elim_candidate) {
                collector_.found_candidate_pair(exposedOffload, *killing_entry);
            }
            it = exposed.erase(it);
            continue;
        }
        ++it;
    }
}


void DataTransferEliminationAnalysis::handle_lib_node(Block& block, data_flow::LibraryNode& node) {
    BaseUserVisitor::handle_lib_node(block, node);

    if (auto* offload_node = dynamic_cast<offloading::DataOffloadingNode*>(&node)) {
        get_or_create_state(block).found_offload_node(block, *offload_node);
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
