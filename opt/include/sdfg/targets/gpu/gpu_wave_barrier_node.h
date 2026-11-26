#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuWaveBarrier("gpu::WaveBarrier");

/**
 * @brief Synchronizes the lanes of a wave (warp) and orders their memory accesses.
 *
 * Executed uniformly by every lane of the wave. Memory writes a lane issued before the node are
 * visible to the other lanes of the wave after it (CUDA `__syncwarp()`, ROCm wavefront-scoped
 * fences around a wave barrier).
 */
class GpuWaveBarrierNode : public data_flow::LibraryNode {
public:
    GpuWaveBarrierNode(
        size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
    );

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class GpuWaveBarrierNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

/// Emits @ref GpuWaveBarrierNode for the CUDA and ROCm kernel language extensions.
class GpuWaveBarrierDispatcher : public codegen::LibraryNodeDispatcher {
public:
    GpuWaveBarrierDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace sdfg::gpu
