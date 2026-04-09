#include "sdfg/analysis/data_transfer_elimination_analysis.h"


#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg::analysis {

OffloadState::OffloadState(DataTransferEliminationCandidateCollector& collector) : collector_(collector) {}

void OffloadState::found_escape(const std::string& container) { kills_containers_.insert(container); }

void OffloadState::found_ptr_write(const std::string& container) { kills_containers_.insert(container); }

void OffloadState::found_ptr_read(const std::string& container) { kills_containers_.insert(container); }

void OffloadState::found_offloaded_kernel(data_flow::LibraryNode& libnode) {
    generated_.clear(); // all generateds are from before and now wiped
    full_kill_ = true;
    // TODO more granularity: find the vars involved in the kernel and put them in kills_containers instead. But full
    // kill should allow simpler testing
    //  can be easier with transfers included in libnode, same as map if not
}

void OffloadState::found_offloaded_kernel(Map& map) {
    generated_.clear(); // all generateds are from before and now wiped
    full_kill_ = true;
    // TODO more granularity: find the vars involved in the kernel and put them in kills_containers instead. But full
    // kill should allow simpler testing
}

std::pair<bool, const OffloadHolder*> OffloadState::find_killing_entry_node(const OffloadHolder& exit_node) const {
    auto& host_access_type = exit_node.host_access->base_type();
    for (auto& entry_node : kernel_entry_nodes_) {
        if (entry_node.host_data->data() == exit_node.host_data->data() &&
            exit_node.node->redundant_with(*entry_node.node)) {
            return {true, &entry_node};
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
        if (dst_is_dev) {
            generated_.emplace(OffloadHolder{&offload, found_host_access, found_host_memlet, found_dev_access});
        } else {
            kernel_entry_nodes_.emplace(OffloadHolder{&offload, found_host_access, found_host_memlet, found_dev_access}
            );
        }
    }
}

void OffloadState::apply_kills(ExposedType& exposed) const {
    if (full_kill_) {
        exposed.clear();
        return;
    }
    for (auto it = exposed.begin(); it != exposed.end(); ++it) {
        auto& holder = *it;
        auto* host = holder.host_data;
        if (host && kills_containers_.contains(host->data())) {
            it = exposed.erase(it);
            continue;
        }
        auto [is_elim_candidate, killing_entry] = find_killing_entry_node(holder);
        if (killing_entry) {
            if (is_elim_candidate) {
                collector_.found_candidate_pair(holder, *killing_entry);
            }
            it = exposed.erase(it);
            continue;
        }
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

    auto* map = dynamic_cast<sdfg::structured_control_flow::Map*>(&loop);

    if (map && map->schedule_type().category() == ScheduleTypeCategory::Offloader) {
        get_or_create_state(loop).found_offloaded_kernel(*map);
    }
}

void DataTransferEliminationAnalysis::
    on_escape(const std::string& container, const ControlFlowNode* node, const Element* user) {
    get_or_create_state(*node).found_escape(container);
}

void DataTransferEliminationAnalysis::
    on_read_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) {
    if (!dynamic_cast<const offloading::DataOffloadingNode*>(&user->dst())) {
        get_or_create_state(*node).found_ptr_read(container);
    }
}

void DataTransferEliminationAnalysis::
    on_write_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) {
    if (!dynamic_cast<const offloading::DataOffloadingNode*>(&user->dst())) {
        get_or_create_state(*node).found_ptr_write(container);
    }
}

std::unique_ptr<OffloadState> DataTransferEliminationAnalysis::
    create_initial_state(const structured_control_flow::ControlFlowNode& node) {
    return std::make_unique<OffloadState>(*this);
}

void DataTransferEliminationAnalysis::run() {
    dispatch(sdfg_.root());

    run_forward(sdfg_.root());

    for (auto& candidate : candidates_) {
        auto& copy_out = candidate.first;
        auto& copy_in = candidate.second;
        DEBUG_PRINTLN(
            "  Eliminating "
            << "copy-out: #" << copy_out.node->element_id() << " " << copy_out.dev_data->data() << " -> "
            << (copy_out.host_data ? copy_out.host_data->data() : "-") << " / copy-in: #" << copy_in.node->element_id()
            << " " << (copy_in.host_data ? copy_in.host_data->data() : "-") << " -> " << copy_in.dev_data->data()
        );
    }
    std::cerr << "Ran analysis" << std::endl;
}

} // namespace sdfg::analysis
