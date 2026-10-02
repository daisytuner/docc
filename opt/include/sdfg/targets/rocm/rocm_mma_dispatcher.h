#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

inline data_flow::ImplementationType ImplementationType_ROCM_MMA_GFX1201("ROCM_MMA_GFX1201");
inline data_flow::ImplementationType ImplementationType_ROCM_MMA_GFX90A("ROCM_MMA_GFX90A");
inline data_flow::ImplementationType ImplementationType_ROCM_MMA("ROCM_MMA");

class RocmMmaBaseDispatcher : public GpuMmaMatmulDispatcher {
protected:
    const GpuMmaSupport* get_mma_arch_from_impl_type_hack() const override;

    const GpuArch* get_gpu_arch_from_context(codegen::CodegenOutput& out) const override;

public:
    RocmMmaBaseDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    )
        : GpuMmaMatmulDispatcher(language_extension, function, data_flow_graph, node) {
    }

protected:
    void emit_block_frag_declaration(
        codegen::CodegenOutput& out,
        const std::string& name,
        MmaFragmentType type,
        std::array<int, 3> dims,
        MmaFragmentLayout layout,
        types::PrimitiveType scalar_type,
        std::optional<std::pair<int, int>> coop_dims
    ) const override;

    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;

    void emit_load_macro(
        codegen::CodegenOutput& out,
        const std::string& name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        MmaFragmentLayout layout
    ) const override;
    void emit_store_macro(
        const codegen::CodegenOutput& out,
        const std::string& frag_name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        MmaFragmentLayout layout
    ) const override;

    void emit_frag_zero_init(codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type)
        const override;
    void emit_mma_compute(
        codegen::CodegenOutput& out,
        const std::string& frag_d_out,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c_in
    ) const override;

    void emit_needed_declarations(codegen::CodegenOutput& out) const override;
};

/// Dispatches a gpu::GpuMmaNode: a single MMA accumulation into a preinitialized accumulator.
class RocmMmaMatmulDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_single_mma(out, inputs);
    }
};

/// Dispatches a gpu::GpuMmaFillNode: declare and zero-initialize an MMA accumulator fragment.
class RocmMmaFillDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_accumulator_fill(out, inputs);
    }
};

/// Dispatches a gpu::MmaFragmentLoadNode: load a tile from memory into a fragment.
class RocmMmaFragmentLoadDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_fragment_load(out, inputs);
    }
};

/// Dispatches a gpu::MmaFragmentStoreNode: store a fragment to a tile in memory.
class RocmMmaFragmentStoreDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_fragment_store(out, inputs);
    }
};

class RocmMmaEltwiseAddDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_eltwise_add(out, inputs);
    }
};

class RocmWmmaLibDependency : public codegen::LibDependency {
public:
    static const RocmWmmaLibDependency* instance() {
        static RocmWmmaLibDependency inst;
        return &inst;
    }

    std::string_view name() const override {
        return "rocwmma";
    }
    void enumerate_includes(std::vector<std::string>& out_list) const override {
        out_list.push_back("rocwmma/rocwmma.hpp");
    }
    std::vector<std::string_view>& globally_unique_ids() const override {
        static std::vector<std::string_view> ids{"rocwmma"};
        return ids;
    }
};

} // namespace sdfg::gpu::rocm
