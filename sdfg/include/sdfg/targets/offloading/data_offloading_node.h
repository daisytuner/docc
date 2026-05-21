#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <sdfg/data_flow/data_flow_graph.h>
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
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

    static std::vector<std::string>
    output_conns(DataTransferDirection transfer_direction, BufferLifecycle buffer_lifecycle);
    static std::vector<std::string> input_conns(DataTransferDirection transfer_direction, BufferLifecycle buffer_lifecycle);

public:
    DataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode code,
        DataTransferDirection transfer_direction,
        BufferLifecycle buffer_lifecycle,
        symbolic::Expression size
    );

    /**
     * Allows to manually override the connector names for subclasses. Subclasses may also need to override other
     * methods to match if order and count of connectors differs
     */
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

    int host_ptr_input_idx() const;
    int dev_ptr_input_idx() const;
    int dev_ptr_output_idx() const;

    const std::string& dev_in_conn() const;
    const std::string& dev_out_conn() const;
    const std::string& host_in_conn() const;

    const symbolic::Expression size() const;

    virtual const symbolic::Expression alloc_size() const;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual symbolic::Expression flop() const override;

    virtual std::string toStr() const override;

    /**
     * Checks that the nodes operate on the same device and compatible data
     */
    virtual bool is_compatible_with(const DataOffloadingNode& other) const;

    /**
     * The 2 nodes are opposites of each other that could cancel each other out
     */
    virtual bool redundant_with(const DataOffloadingNode& other) const;

    /**
     * The 2 nodes do the same thing
     */
    virtual bool equal_with(const DataOffloadingNode& other) const;

    virtual bool is_same_target(const DataOffloadingNode& other) const = 0;

    virtual bool blocking() const = 0;

    bool is_d2h() const;
    bool is_h2d() const;
    bool has_transfer() const;

    bool is_free() const;
    bool is_alloc() const;

    void remove_free();

    void remove_h2d();
    void remove_d2h();

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    data_flow::EdgeRemoveOption can_remove_out_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet)
        const override;

    data_flow::EdgeRemoveOption can_remove_in_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet)
        const override;

    bool update_edge_removed(const std::string& out_conn) override;
};

template<typename NodeT, typename... Args>
data_flow::LibraryNode& add_offloading_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& host_container,
    const std::string& dev_container,
    DataTransferDirection direction,
    BufferLifecycle lifecycle,
    const types::IType& host_type,
    const types::IType& dev_type,
    DebugInfo debug_info = DebugInfo(),
    Args&&... args
) {
    std::optional<std::string> host_input_container;
    std::optional<std::string> dev_input_container;
    std::optional<std::string> output_container;

    if (is_H2D(direction) && !is_ALLOC(lifecycle)) {
        host_input_container = host_container;
        dev_input_container = dev_container;
    } else if (is_H2D(direction) && is_ALLOC(lifecycle)) {
        host_input_container = host_container;
    } else if (!is_NONE(direction)) {
        host_input_container = host_container;
        dev_input_container = dev_container;
    } else if (is_FREE(lifecycle)) {
        dev_input_container = dev_container;
    }

    if (is_ALLOC(lifecycle)) {
        output_container = dev_container;
    }

    data_flow::AccessNode* host_in_access = nullptr;
    if (host_input_container) {
        host_in_access = &builder.add_access(block, host_input_container.value());
    }
    data_flow::AccessNode* dev_in_access = nullptr;
    if (dev_input_container) {
        dev_in_access = &builder.add_access(block, dev_input_container.value());
    }
    data_flow::AccessNode* out_access = nullptr;
    if (output_container) {
        out_access = &builder.add_access(block, output_container.value());
    }
    auto& libnode =
        builder.template add_library_node<NodeT>(block, debug_info, direction, lifecycle, std::forward<Args>(args)...);

    if (host_in_access) {
        builder.add_computational_memlet(block, *host_in_access, libnode, "_hst", {}, host_type);
    }
    if (dev_in_access) {
        builder.add_computational_memlet(block, *dev_in_access, libnode, "_dev", {}, dev_type);
    }
    if (out_access) {
        builder.add_computational_memlet(block, libnode, "_dev", *out_access, {}, dev_type);
    }

    return libnode;
}

template<typename NodeT, typename... Args>
std::tuple<structured_control_flow::Block&, data_flow::LibraryNode&> add_offloading_block(
    builder::StructuredSDFGBuilder& builder,
    Sequence& parent,
    const std::string& host_container,
    const std::string& dev_container,
    DataTransferDirection direction,
    BufferLifecycle lifecycle,
    const types::IType& data_type,
    DebugInfo debug_info = DebugInfo(),
    Args&&... args
) {
    auto& block = builder.add_block(parent);

    auto& libnode = add_offloading_node<NodeT>(
        builder,
        block,
        host_container,
        dev_container,
        direction,
        lifecycle,
        data_type,
        data_type,
        debug_info,
        std::forward<Args>(args)...
    );

    return {block, libnode};
}

class DataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    template<typename NodeT, typename BuilderT, typename... Args>
    static data_flow::LibraryNode& deserialize_generic_offload(
        const nlohmann::json& j, BuilderT& builder, structured_control_flow::Block& parent, Args&&... args
    ) {
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        symbolic::Expression size;
        if (!j.contains("size") || j.at("size").is_null()) {
            size = SymEngine::null;
        } else {
            size = symbolic::parse(j.at("size"));
        }
        auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
        auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());

        return builder.template add_library_node<
            NodeT>(parent, debug_info, transfer_direction, buffer_lifecycle, size, std::forward<Args>(args)...);
    }
};

} // namespace offloading
} // namespace sdfg
