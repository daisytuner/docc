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

constexpr bool dump_offload_node_ids = true;

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

DataOffloadingNode::DataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode code,
    DataTransferDirection transfer_direction,
    BufferLifecycle buffer_lifecycle,
    symbolic::Expression size
)
    : DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          code,
          output_conns(transfer_direction, buffer_lifecycle),
          input_conns(transfer_direction, buffer_lifecycle),
          transfer_direction,
          buffer_lifecycle,
          size
      ) {}

std::vector<std::string> DataOffloadingNode::
    output_conns(DataTransferDirection transfer_direction, BufferLifecycle buffer_lifecycle) {
    if (is_ALLOC(buffer_lifecycle)) {
        return {"_dev"};
    } else {
        return {};
    }
}

std::vector<std::string> DataOffloadingNode::
    input_conns(DataTransferDirection transfer_direction, BufferLifecycle buffer_lifecycle) {
    if (is_H2D(transfer_direction) && is_ALLOC(buffer_lifecycle)) {
        return {"_hst"};
    } else if (!is_NONE(transfer_direction)) {
        return {"_hst", "_dev"};
    } else if (is_FREE(buffer_lifecycle)) {
        return {"_dev"};
    } else {
        return {};
    }
}

int DataOffloadingNode::dev_ptr_input_idx() const {
    if (transfer_direction_ == DataTransferDirection::NONE && buffer_lifecycle_ == BufferLifecycle::FREE) {
        return 0;
    } else if (transfer_direction_ == DataTransferDirection::D2H) {
        return 1;
    } else if (transfer_direction_ == DataTransferDirection::H2D && buffer_lifecycle_ != BufferLifecycle::ALLOC) {
        return 1;
    } else {
        return -1;
    }
}

int DataOffloadingNode::host_ptr_input_idx() const {
    if (transfer_direction_ != DataTransferDirection::NONE) {
        return 0;
    } else {
        return -1;
    }
}

int DataOffloadingNode::dev_ptr_output_idx() const {
    if (buffer_lifecycle_ == BufferLifecycle::ALLOC) {
        return 0;
    } else {
        return -1;
    }
}

const std::string& DataOffloadingNode::dev_in_conn() const { return inputs_.at(dev_ptr_input_idx()); }

const std::string& DataOffloadingNode::dev_out_conn() const { return outputs_.at(dev_ptr_output_idx()); }

const std::string& DataOffloadingNode::host_in_conn() const { return inputs_.at(host_ptr_input_idx()); }

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

bool DataOffloadingNode::is_compatible_with(const DataOffloadingNode& other) const {
    if (code() != other.code()) {
        return false;
    }
    if (!symbolic::null_safe_eq(size(), other.size())) {
        return false;
    }
    return true;
}

bool DataOffloadingNode::redundant_with(const DataOffloadingNode& other) const {
    if (!is_compatible_with(other)) {
        return false;
    }
    if ((static_cast<int8_t>(transfer_direction()) + static_cast<int8_t>(other.transfer_direction())) != 0) {
        return false; // not the inverse
    }
    if ((static_cast<int8_t>(buffer_lifecycle()) + static_cast<int8_t>(other.buffer_lifecycle())) != 0) {
        return false;
    }

    return true; // add more checks in sub-classes
}

bool DataOffloadingNode::equal_with(const DataOffloadingNode& other) const {
    if (!is_compatible_with(other)) {
        return false;
    }
    if (this->transfer_direction() != other.transfer_direction()) {
        return false;
    }
    if (this->buffer_lifecycle() != other.buffer_lifecycle()) {
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
    if (is_h2d() && input_idx == host_ptr_input_idx()) {
        return data_flow::PointerAccessMeta::create_read_only(size_, true);
    } else if (is_h2d() && !is_alloc() && input_idx == dev_ptr_input_idx()) {
        return data_flow::PointerAccessMeta::create_full_write_only(size_, true);
    } else if (is_d2h() && input_idx == dev_ptr_input_idx()) {
        return data_flow::PointerAccessMeta::create_read_only(size_, true);
    } else if (is_d2h() && input_idx == host_ptr_input_idx()) {
        return data_flow::PointerAccessMeta::create_full_write_only(size_, true);
    } else if (is_d2h() && is_free() && input_idx == dev_ptr_input_idx()) {
        return data_flow::PointerAccessMeta::create_invalidate();
    } else if (is_d2h() && !is_free() && input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(size_, true);
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

void DataOffloadingNode::remove_d2h() {
    if (this->is_d2h()) {
        if (!this->is_free()) {
            throw InvalidSDFGException("DataOffloadingNode: Tried removing d2h but node has no other purpose");
        }
        this->transfer_direction_ = DataTransferDirection::NONE;
        this->inputs_.erase(this->inputs_.begin());
    }
}

data_flow::EdgeRemoveOption DataOffloadingNode::
    can_remove_out_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet) const {
    if (graph.out_edges_for_connector(*this, memlet->src_conn()).size() > 1) {
        return data_flow::EdgeRemoveOption::Trivially;
    } else if (is_alloc() && outputs_.size() == 1 && memlet->src_conn() == outputs_.at(0)) {
        // the node in its entirety is dead if it the alloc is not needed
        return data_flow::EdgeRemoveOption::RemoveNodeAfter;
    } else {
        return data_flow::EdgeRemoveOption::NotRemovable;
    }
}

bool DataOffloadingNode::update_edge_removed(const std::string& out_conn) { return false; }

data_flow::EdgeRemoveOption DataOffloadingNode::
    can_remove_in_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet) const {
    if (is_h2d() && is_alloc() && memlet->dst_conn() == inputs_.at(host_ptr_input_idx())) {
        return data_flow::EdgeRemoveOption::RequiresUpdate;
    } else if (is_d2h() && is_NO_CHANGE(this->buffer_lifecycle_) &&
               memlet->dst_conn() == inputs_.at(host_ptr_input_idx())) {
        return data_flow::EdgeRemoveOption::RemoveNodeAfter;
    } else {
        return data_flow::EdgeRemoveOption::NotRemovable;
    }
}

nlohmann::json DataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const DataOffloadingNode&>(library_node);
    nlohmann::json j;

    // Library node properties
    j["code"] = std::string(library_node.code().value());

    // Offloading node properties
    sdfg::serializer::JSONSerializer serializer;
    if (node.size().is_null()) {
        j["size"] = nlohmann::json::value_t::null;
    } else {
        j["size"] = serializer.expression(node.size());
    }
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

} // namespace offloading
} // namespace sdfg
