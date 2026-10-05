#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"
#include "sdfg/targets/gpu/math/tensor/tensor_operands.h"

namespace sdfg::gpu::tensor {

/**
 * @brief Base for library node dispatchers that run a tensor node as hand-written GPU kernels.
 *
 * Handles everything that is not layer specific: the host<->device transfers of
 * the `WithTransfers` variant (driven by @ref tensor_operands), the kernel source
 * files and the kernel launch. Backend differences come from the
 * @ref GPUOffloadDispatcherStrategy. Subclasses only emit their kernels.
 */
class GPUTensorNodeDispatcher : public codegen::LibraryNodeDispatcher {
protected:
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy_;
    bool with_transfers_;

    static constexpr int BLOCK_SIZE = 256;
    static constexpr int MAX_GRID_SIZE = 65535;

    /// Emit the kernels and their launches; `device_ptrs` maps every input connector to a device pointer.
    virtual void dispatch_device(
        codegen::CodegenOutput& out,
        const std::vector<TensorOperand>& operands,
        const std::unordered_map<std::string, std::string>& device_ptrs
    ) = 0;

    /// Create the kernel translation unit and its include header; returns the kernel source stream.
    codegen::PrettyPrinter& require_kernel_source(codegen::CodegenOutput& out, const std::string& kernel_name);

    /// Launch a 1D grid-stride kernel covering `num_work_items` (a host expression); skipped when empty.
    void dispatch_launch(
        codegen::CodegenOutput& out,
        const std::string& kernel_name,
        const std::string& num_work_items,
        std::vector<std::string> arguments
    );

    std::string kernel_name(const std::string& prefix) const;

public:
    GPUTensorNodeDispatcher(
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

/// Unsigned integer type of the same width as `type`, used for bit-exact element copies.
std::string copy_type(types::PrimitiveType type);

/// C type of an integer index element.
std::string index_type(types::PrimitiveType type);

} // namespace sdfg::gpu::tensor
