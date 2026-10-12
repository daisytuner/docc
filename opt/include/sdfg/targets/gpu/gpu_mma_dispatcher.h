#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

/**
 * @brief Hardware-dependent emit primitives for the composable MMA library nodes.
 *
 * A GpuMma*Dispatcher owns one strategy and delegates only the parts that differ between targets
 * (CUDA vs ROCm) and implementations (fragment/WMMA vs MFMA) to it. Everything reusable - reading
 * the nodes, wiring up operands - stays in the dispatchers.
 */
class GpuMmaDispatcherStrategy {
public:
    /// Per-element callback used by @ref emit_eltwise: writes one output element from its inputs.
    using EltwiseCompute = std::function<void(
        codegen::CodegenOutput& out,
        const std::string& main_frag_elem,
        const std::string& idx,
        const std::vector<std::string>& other_frag_elems
    )>;

    virtual ~GpuMmaDispatcherStrategy() = default;

    /// Target library includes / helper declarations needed before the first emit.
    virtual void emit_needed_declarations(codegen::CodegenOutput& out) const {
    }

    /// Zero-initialize the accumulator fragment behind @p acc_frag.
    virtual void
    emit_fill(codegen::CodegenOutput& out, const std::string& acc_frag, types::PrimitiveType fill_type) const = 0;

    /// frag_d = frag_a * frag_b + frag_c (single MMA step; all operands are fragments).
    virtual void emit_mma(
        codegen::CodegenOutput& out,
        const std::string& frag_d,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c
    ) const = 0;

    /// Load a tile from memory (base + layout) into the fragment behind @p frag.
    virtual void emit_fragment_load(
        codegen::CodegenOutput& out,
        const std::string& frag,
        const std::string& base,
        const GpuMmaFromMemoryLayout& layout,
        MmaFragmentType frag_type,
        types::PrimitiveType element_type
    ) const = 0;

    /// Store the fragment behind @p frag to memory (base + layout).
    virtual void emit_fragment_store(
        codegen::CodegenOutput& out,
        const std::string& frag,
        const std::string& base,
        const GpuMmaFromMemoryLayout& layout,
        MmaFragmentType frag_type,
        types::PrimitiveType element_type
    ) const = 0;

    /// Element-wise loop over @p main_frag, invoking @p compute for each element.
    virtual void emit_eltwise(
        codegen::CodegenOutput& out,
        const std::string& main_frag,
        const std::vector<std::string>& additional_frags,
        const EltwiseCompute& compute
    ) const = 0;
};

/// Shared storage for the strategy; each composable MMA node has its own dispatcher subclass.
class GpuMmaNodeDispatcher : public codegen::LibraryNodeDispatcher {
protected:
    std::unique_ptr<GpuMmaDispatcherStrategy> strategy_;

public:
    GpuMmaNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node,
        std::unique_ptr<GpuMmaDispatcherStrategy> strategy
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), strategy_(std::move(strategy)) {
    }
};

/// A single MMA accumulation of A * B into a preinitialized accumulator (gpu::GpuMmaMatmul).
class GpuMmaMatmulDispatcher : public GpuMmaNodeDispatcher {
public:
    using GpuMmaNodeDispatcher::GpuMmaNodeDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/// Declare and zero-initialize an MMA accumulator fragment (gpu::GpuMmaFill).
class GpuMmaFillDispatcher : public GpuMmaNodeDispatcher {
public:
    using GpuMmaNodeDispatcher::GpuMmaNodeDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/// Load a tile from memory into a fragment (gpu::GpuMmaFragmentLoad).
class GpuMmaFragmentLoadDispatcher : public GpuMmaNodeDispatcher {
public:
    using GpuMmaNodeDispatcher::GpuMmaNodeDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/// Store a fragment to a tile in memory (gpu::GpuMmaFragmentStore).
class GpuMmaFragmentStoreDispatcher : public GpuMmaNodeDispatcher {
public:
    using GpuMmaNodeDispatcher::GpuMmaNodeDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/// D = out_type(Acc + acc_type(C)), element-wise over the fragments (gpu::GpuMmaFragmentEltwiseAdd).
class GpuMmaEltwiseAddDispatcher : public GpuMmaNodeDispatcher {
public:
    using GpuMmaNodeDispatcher::GpuMmaNodeDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::gpu
