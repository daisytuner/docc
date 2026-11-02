#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

inline data_flow::ImplementationType ImplementationType_ROCM_MMA_FRAGMENT("ROCM_MMA");

/// Classic rocwmma fragment-based emit strategy (load/store/mma_sync/fill_fragment).
class RocmMmaFragmentStrategy : public GpuMmaDispatcherStrategy {
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

    std::string fragment_element(const std::string& frag, const std::string& idx) const override {
        return frag + ".x[" + idx + "]";
    }

    std::string fragment_element_count(const std::string& frag) const override {
        return "static_cast<int>(" + frag + ".num_elements)";
    }

    /// On wave64 targets (per the queried wavefront size), 16x16 accumulators keep the MFMA output layout:
    /// lane l holds column l % 16, rows 4 * (l / 16) + e. Other wavefront sizes fall back to the iota path.
    std::optional<std::pair<std::string, std::string>> fragment_element_coords(
        MmaFragmentType frag_type, const MmaBlockSize& block_size, const std::string& idx, std::string& condition
    ) const override;
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
