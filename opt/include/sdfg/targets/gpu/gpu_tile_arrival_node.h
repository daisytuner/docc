#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuTileArrival("gpu::TileArrival");

/**
 * @brief Counts the arrival of a thread block at a shared work item and elects the last arriving block.
 *
 * The single input `counter` points to a device array of arrival counters that are zero before the first
 * arrival (e.g. a @ref types::StorageType::GPU_Static buffer). The node is executed uniformly by all threads
 * of a block and performs, for the counter at @ref index:
 *
 *  1. a device-wide fence and a block barrier, so every global write the block issued before the node is
 *     visible to the other blocks once it has arrived;
 *  2. one atomic increment of `counter[index]` by a single thread of the block;
 *  3. a block-wide broadcast of whether this increment was the @ref expected -th one.
 *
 * The output `_last` is 1 in every thread of the electing block and 0 in every thread of the other blocks.
 * The electing block issues a device-wide fence after the election, so it observes all writes the other
 * blocks made before their arrival, and resets `counter[index]` to zero for the next use.
 */
class GpuTileArrivalNode : public data_flow::LibraryNode {
    symbolic::Expression index_;
    symbolic::Expression expected_;

public:
    static constexpr auto COUNTER_INPUT_IDX = 0;

    GpuTileArrivalNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::Expression& index,
        const symbolic::Expression& expected
    );

    /// Element of the counter array the block arrives at.
    const symbolic::Expression& index() const {
        return index_;
    }

    /// Number of arrivals that completes the work item.
    const symbolic::Expression& expected() const {
        return expected_;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class GpuTileArrivalNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

/// Emits @ref GpuTileArrivalNode for the CUDA and ROCm kernel language extensions.
class GpuTileArrivalDispatcher : public codegen::LibraryNodeDispatcher {
    std::string flag_;

public:
    GpuTileArrivalDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::gpu
