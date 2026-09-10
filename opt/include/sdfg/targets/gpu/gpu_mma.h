#pragma once

#include <sdfg/passes/expansion/lib_node_expander.h>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"

namespace sdfg::gpu {

class GpuMmaExpander : public passes::CodeLibNodeExpander<math::tensor::MatMulNode> {
protected:

public:
    GpuMmaExpander() : CodeLibNodeExpander(math::tensor::LibraryNodeType_MatMul) {}
    virtual ~GpuMmaExpander() = default;
    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override;

protected:
    virtual bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const = 0;
};

class GpuMmaMatmulDispatcher : public codegen::LibraryNodeDispatcher {
public:
    enum class FragmentType { A, B, C };

    struct MmaTiling {
        int mma_block_m = 0;
        int mma_block_n = 0;
        int mma_block_k = 0;
        int wave_tile_blocks_m = 0;
        int wave_tile_blocks_n = 0;
        int macro_blocks_m = 0;
        int macro_blocks_n = 0;
    };

protected:
    virtual const GpuMmaSupport* get_mma_arch() const = 0;

    virtual MmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const = 0;

    virtual void emit_block_frag_declaration(
        codegen::CodegenOutput& out,
        const std::string& name,
        FragmentType type,
        std::array<int, 3> dims,
        math::tensor::TensorLayout::TensorLayoutType layout,
        types::PrimitiveType scalar_type,
        std::optional<std::pair<int, int>> coop_dims = std::nullopt
    ) const = 0;

    virtual void emit_load_macro(
        codegen::CodegenOutput& out,
        const std::string& name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        math::tensor::TensorLayout::TensorLayoutType layout = math::tensor::TensorLayout::LAYOUT_OTHER
    ) const = 0;
    virtual void emit_store_macro(
        const codegen::CodegenOutput& out,
        const std::string& frag_name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        math::tensor::TensorLayout::TensorLayoutType layout = math::tensor::TensorLayout::LAYOUT_OTHER
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

public:
    GpuMmaMatmulDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}


    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;

    static bool is_col_major(const math::tensor::TensorLayout& layout);
};

} // namespace sdfg::gpu
