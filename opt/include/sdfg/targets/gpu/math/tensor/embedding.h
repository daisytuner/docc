#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"

namespace sdfg::gpu::tensor {

/// Bytes behind the embedding's `Y`, `W` or `I` connector.
symbolic::Expression embedding_bytes(
    const math::tensor::EmbeddingNode& node, const data_flow::DataFlowGraph& graph, const std::string& connector
);

/**
 * @brief Gathers `Y[b..., j] = W[I[b...], j]` on the GPU, one thread per output element (grid-stride).
 *
 * Backend differences (launch, kernel files, runtime API) come from the
 * @ref GPUOffloadDispatcherStrategy. With transfers, `W` and `I` are copied to the
 * device and `Y` is copied back; without, all pointers are already device pointers.
 */
class GPUEmbeddingDispatcher : public codegen::LibraryNodeDispatcher {
private:
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy_;
    bool with_transfers_;

    void dispatch_kernel(codegen::CodegenOutput& out, const std::string& y, const std::string& w, const std::string& i);

public:
    GPUEmbeddingDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node,
        std::unique_ptr<GPUOffloadDispatcherStrategy> strategy,
        bool with_transfers
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

/// Register the GPU embedding dispatcher for one backend, under its with/without-transfers implementation types.
template<typename Strategy>
void register_gpu_embedding_dispatchers(
    codegen::LibraryNodeDispatcherRegistry& registry,
    const data_flow::ImplementationType& with_transfers,
    const data_flow::ImplementationType& without_transfers
) {
    for (bool transfers : {true, false}) {
        registry.register_library_node_dispatcher(
            math::tensor::LibraryNodeType_Embedding,
            transfers ? with_transfers : without_transfers,
            [transfers](
                codegen::LanguageExtension& language_extension,
                const Function& function,
                const data_flow::DataFlowGraph& data_flow_graph,
                const data_flow::LibraryNode& node
            ) {
                // The strategy's kernel language extension needs a mutable SDFG.
                auto& sdfg = const_cast<StructuredSDFG&>(dynamic_cast<const StructuredSDFG&>(function));
                return std::make_unique<GPUEmbeddingDispatcher>(
                    language_extension, function, data_flow_graph, node, std::make_unique<Strategy>(sdfg), transfers
                );
            }
        );
    }
}

} // namespace sdfg::gpu::tensor
