#pragma once

#include <memory>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/gpu/math/tensor/gpu_tensor_dispatcher.h"

namespace sdfg::gpu::tensor {

/// Gathers `Y[b..., j] = W[I[b...], j]` with one thread per output element (grid-stride).
class GPUEmbeddingDispatcher : public GPUTensorNodeDispatcher {
protected:
    void dispatch_device(
        codegen::CodegenOutput& out,
        const std::vector<TensorOperand>& operands,
        const std::unordered_map<std::string, std::string>& device_ptrs
    ) override;

public:
    using GPUTensorNodeDispatcher::GPUTensorNodeDispatcher;
};

/// Register every GPU tensor dispatcher for one backend, under its with/without-transfers implementation types.
template<typename Strategy>
void register_gpu_tensor_dispatchers(
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
