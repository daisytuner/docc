#include "sdfg/targets/offloading/data_offloading_node.h"

#include <cstddef>
#include <string>
#include <vector>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace offloading {

constexpr bool dump_offload_node_ids = false;

DataOffloadingNode::DataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    DataTransferDirection transfer_direction,
    BufferLifecycle buffer_lifecycle,
    symbolic::Expression size
)
    : data_flow::LibraryNode(
          element_id, debug_info, vertex, parent, code, outputs, inputs, true, data_flow::ImplementationType_NONE
      ),
      transfer_direction_(transfer_direction), buffer_lifecycle_(buffer_lifecycle), size_(std::move(size)) {}

DataTransferDirection DataOffloadingNode::transfer_direction() const { return this->transfer_direction_; }

BufferLifecycle DataOffloadingNode::buffer_lifecycle() const { return this->buffer_lifecycle_; }

const symbolic::Expression DataOffloadingNode::size() const { return this->size_; }

const symbolic::Expression DataOffloadingNode::alloc_size() const { return this->size(); }

symbolic::SymbolSet DataOffloadingNode::symbols() const {
    if (this->size().is_null()) {
        return {};
    } else {
        return symbolic::atoms(this->size());
    }
}

void DataOffloadingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (!this->size_.is_null()) {
        this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
    }
}

std::string DataOffloadingNode::toStr() const {
    std::string direction, lifecycle;
    switch (this->transfer_direction()) {
        case DataTransferDirection::D2H:
            direction = " D2H";
            break;
        case DataTransferDirection::H2D:
            direction = " H2D";
            break;
        default:
            direction = " NONE";
            break;
    }
    switch (this->buffer_lifecycle()) {
        case BufferLifecycle::FREE:
            lifecycle = " FREE";
            break;
        case BufferLifecycle::ALLOC:
            lifecycle = " ALLOC";
            break;
        default:
            lifecycle = " NO_CHANGE";
            break;
    }
    std::string res = std::string(this->code_.value());
    if (dump_offload_node_ids) {
        res += " #" + std::to_string(element_id_);
    }
    res += direction + lifecycle;
    return res;
}

symbolic::Expression DataOffloadingNode::flop() const { return symbolic::zero(); }

bool DataOffloadingNode::redundant_with(const DataOffloadingNode& other) const {
    if (code() != other.code()) {
        return false;
    }
    if ((static_cast<int8_t>(transfer_direction()) + static_cast<int8_t>(other.transfer_direction())) != 0) {
        return false; // not the inverse
    }
    if ((static_cast<int8_t>(buffer_lifecycle()) + static_cast<int8_t>(other.buffer_lifecycle())) != 0) {
        return false;
    }

    if (!symbolic::null_safe_eq(size(), other.size())) {
        return false;
    }

    return true; // add more checks in sub-classes
}

bool DataOffloadingNode::equal_with(const DataOffloadingNode& other) const {
    if (code() != other.code()) {
        return false;
    }
    if (this->transfer_direction() != other.transfer_direction()) {
        return false;
    }
    if (this->buffer_lifecycle() != other.buffer_lifecycle()) {
        return false;
    }

    if (!symbolic::null_safe_eq(size(), other.size())) {
        return false;
    }

    return true; // add more checks in sub-classes
}

bool DataOffloadingNode::is_d2h() const { return is_D2H(this->transfer_direction()); }

bool DataOffloadingNode::is_h2d() const { return is_H2D(this->transfer_direction()); }

bool DataOffloadingNode::has_transfer() const { return this->is_d2h() || this->is_h2d(); }

bool DataOffloadingNode::is_free() const { return is_FREE(this->buffer_lifecycle()); }

bool DataOffloadingNode::is_alloc() const { return is_ALLOC(this->buffer_lifecycle()); }

void DataOffloadingNode::remove_h2d() {
    if (this->is_h2d()) {
        if (!this->is_alloc()) {
            throw InvalidSDFGException("DataOffloadingNode: Tried removing h2d but node has no other purpose");
        }
        this->transfer_direction_ = DataTransferDirection::NONE;
        this->inputs_.erase(this->inputs_.begin()); // Standard nodes only have one, others need to override
    }
}

data_flow::PointerAccessType DataOffloadingNode::pointer_access_type(int input_idx) const {
    if (is_h2d() && input_idx == 0) {
        return data_flow::PointerReadOnly(size_, true);
    } else {
        return LibraryNode::pointer_access_type(input_idx);
    }
}

void DataOffloadingNode::remove_free() {
    if (this->is_free()) {
        if (!this->has_transfer()) {
            throw InvalidSDFGException("DataOffloadingNode: Tried removing free but no data transfer direction present"
            );
        }
        this->buffer_lifecycle_ = BufferLifecycle::NO_CHANGE;
    }
}

data_flow::EdgeRemoveOption DataOffloadingNode::
    can_remove_out_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet) const {
    if (graph.out_edges_for_connector(*this, memlet->src_conn()).size() > 1) {
        return data_flow::EdgeRemoveOption::Trivially;
    } else if (transfer_direction_ != DataTransferDirection::NONE && outputs_.size() == 1 &&
               memlet->src_conn() == outputs_.at(0)) {
        // the node represents a transfer, whose output is dead.
        if (buffer_lifecycle_ != BufferLifecycle::NO_CHANGE) {
            // the node still has remaining purpose without the transfer
            return data_flow::EdgeRemoveOption::RequiresUpdate;
        } else {
            // the node in its entirety is dead if it the transfer is not needed
            return data_flow::EdgeRemoveOption::RemoveNodeAfter;
        }
    } else {
        return data_flow::EdgeRemoveOption::NotRemovable;
    }
}

bool DataOffloadingNode::update_edge_removed(const std::string& out_conn) {
    if (transfer_direction_ != DataTransferDirection::NONE && outputs_.size() == 1 && out_conn == outputs_.at(0)) {
        transfer_direction_ = DataTransferDirection::NONE;
        outputs_.erase(outputs_.begin());
        return true;
    } else {
        return false;
    }
}

} // namespace offloading
} // namespace sdfg
