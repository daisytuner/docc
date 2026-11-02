#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

namespace sdfg::gpu::cuda {

inline data_flow::ImplementationType ImplementationType_CUDA_MMA_FRAGMENT("CUDA_MMA");

/// Classic rocwmma fragment-based emit strategy (load/store/mma_sync/fill_fragment).
class CudaMmaFragmentStrategy : public GpuMmaDispatcherStrategy {
public:
    void emit_needed_declarations(codegen::CodegenOutput& out) const override;

    void emit_fill(codegen::CodegenOutput& out, const std::string& acc_frag, types::PrimitiveType fill_type) const override;

    void emit_mma(
        codegen::CodegenOutput& out,
        const std::string& frag_d,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c
    ) const override;

    void emit_fragment_load(
        codegen::CodegenOutput& out,
        const std::string& frag,
        const std::string& base,
        const GpuMmaFromMemoryLayout& layout,
        MmaFragmentType frag_type,
        types::PrimitiveType element_type,
        MmaFragmentLayout fragment_layout
    ) const override;

    void emit_fragment_store(
        codegen::CodegenOutput& out,
        const std::string& frag,
        const std::string& base,
        const GpuMmaFromMemoryLayout& layout,
        MmaFragmentType frag_type,
        types::PrimitiveType element_type
    ) const override;

    void emit_eltwise(
        codegen::CodegenOutput& out,
        const std::string& main_frag,
        const std::vector<std::string>& additional_frags,
        const EltwiseCompute& compute
    ) const override;
};

class CudaWmmaLibDependency : public codegen::LibDependency {
public:
    static const CudaWmmaLibDependency* instance() {
        static CudaWmmaLibDependency inst;
        return &inst;
    }

    std::string_view name() const override {
        return "nvcuda-wmma";
    }
    void enumerate_includes(std::vector<std::string>& out_list) const override {
        out_list.push_back("mma.h");
    }
    std::vector<std::string_view>& globally_unique_ids() const override {
        static std::vector<std::string_view> ids{"nvcuda-wmma"};
        return ids;
    }
};

} // namespace sdfg::gpu::cuda
