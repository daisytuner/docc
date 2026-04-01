#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <sdfg/data_flow/data_flow_graph.h>
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace offloading {

enum class DataTransferDirection : int8_t { D2H = -1, NONE = 0, H2D = 1 };

constexpr bool is_D2H(const DataTransferDirection& transfer_direction) {
    return (transfer_direction == DataTransferDirection::D2H);
}

constexpr bool is_NONE(const DataTransferDirection& transfer_direction) {
    return (transfer_direction == DataTransferDirection::NONE);
}

constexpr bool is_H2D(const DataTransferDirection& transfer_direction) {
    return (transfer_direction == DataTransferDirection::H2D);
}

enum class BufferLifecycle : int8_t { FREE = -1, NO_CHANGE = 0, ALLOC = 1 };

constexpr bool is_FREE(const BufferLifecycle& buffer_lifecycle) { return (buffer_lifecycle == BufferLifecycle::FREE); }

constexpr bool is_NO_CHANGE(const BufferLifecycle& buffer_lifecycle) {
    return (buffer_lifecycle == BufferLifecycle::NO_CHANGE);
}

constexpr bool is_ALLOC(const BufferLifecycle& buffer_lifecycle) {
    return (buffer_lifecycle == BufferLifecycle::ALLOC);
}

/**
 * Name does not match current function. Specific to Offloaded buffer handling, where copies of original data are
 * introduced or the canonical data is moved around
 */
class DataOffloadingNode : public data_flow::LibraryNode {
protected:
    DataTransferDirection transfer_direction_;
    BufferLifecycle buffer_lifecycle_;
    /// In Bytes
    symbolic::Expression size_;

public:
    DataOffloadingNode(
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
    );

    DataTransferDirection transfer_direction() const;
    BufferLifecycle buffer_lifecycle() const;

    const symbolic::Expression size() const;

    virtual const symbolic::Expression alloc_size() const;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual symbolic::Expression flop() const override;

    virtual std::string toStr() const override;

    virtual bool redundant_with(const DataOffloadingNode& other) const;

    virtual bool equal_with(const DataOffloadingNode& other) const;

    virtual bool is_same_target(const DataOffloadingNode& other) const = 0;

    virtual bool blocking() const = 0;

    bool is_d2h() const;
    bool is_h2d() const;
    bool has_transfer() const;

    bool is_free() const;
    bool is_alloc() const;

    void remove_free();
};

} // namespace offloading
} // namespace sdfg
