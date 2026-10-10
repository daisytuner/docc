#pragma once

#include <sdfg/passes/expansion/lib_node_expander.h>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/targets/gpu/gpu_mma_expander.h"

namespace sdfg::gpu {

class GpuMmaMatmulDispatcher : public codegen::LibraryNodeDispatcher {
protected:
    virtual const GpuArch* get_gpu_arch_from_context(codegen::CodegenOutput& out) const = 0;

    virtual void emit_block_frag_declaration(
        codegen::CodegenOutput& out,
        const std::string& name,
        MmaFragmentType type,
        std::array<int, 3> dims,
        MmaFragmentLayout layout,
        types::PrimitiveType scalar_type,
        std::optional<std::pair<int, int>> coop_dims = std::nullopt
    ) const = 0;

    virtual void emit_load_macro(
        codegen::CodegenOutput& out,
        const std::string& name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        MmaFragmentLayout layout = MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
    ) const = 0;
    virtual void emit_store_macro(
        const codegen::CodegenOutput& out,
        const std::string& frag_name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        MmaFragmentLayout layout = MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
    ) const = 0;

    symbolic::Expression get_start_offset(const math::tensor::TensorLayout& layout) const;

    virtual void emit_frag_zero_init(
        codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type
    ) const = 0;
    virtual void emit_mma_compute(
        codegen::CodegenOutput& out,
        const std::string& frag_d_out,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c_in
    ) const = 0;
    virtual void emit_eltwise_compute(
        codegen::CodegenOutput& out,
        const std::string& main_frag,
        const std::vector<std::string>& additional_frags,
        const std::function<void(
            codegen::CodegenOutput&,
            const std::string& main_frag_elem,
            const std::string& idx,
            const std::vector<std::string>& other_frag_elems
        )>& compute
    ) const;

    virtual void emit_needed_declarations(codegen::CodegenOutput& out) const {
    }

    /// Emit a single MMA accumulation of A * B into the preinitialized accumulator behind Y
    /// (used by GpuMmaFragmentMatmulNode). Operates purely on fragments; no memory access.
    void dispatch_single_mma(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const;

    /// Emit code that declares and zero-initializes the MMA accumulator fragment behind Y
    /// (used by GpuMmaFillNode).
    void dispatch_accumulator_fill(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const;

    /// Emit a fragment load from memory into the fragment behind "frag" (used by MmaFragmentLoadNode).
    void dispatch_fragment_load(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const;

    /// Emit a fragment store to memory from the fragment behind "frag" (used by MmaFragmentStoreNode).
    void dispatch_fragment_store(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const;

    /// Emit code to add any input of C together with the accumulator fragment together with correct types
    void dispatch_eltwise_add(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const;

public:
    GpuMmaMatmulDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
    }


    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;

    static bool is_col_major(const math::tensor::TensorLayout& layout);
};

} // namespace sdfg::gpu
