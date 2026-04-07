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
    return std::string(this->code_.value()) + direction + lifecycle;
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

void DataOffloadingNode::remove_free() {
    if (this->is_free()) {
        if (!this->has_transfer()) {
            throw InvalidSDFGException("DataOffloadingNode: Tried removing free but no data transfer direction present"
            );
        }
        this->buffer_lifecycle_ = BufferLifecycle::NO_CHANGE;
    }
}

} // namespace offloading
} // namespace sdfg
