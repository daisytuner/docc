#pragma once

#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

inline data_flow::ImplementationType ImplementationType_ROCM_MMA_BARE("ROCM_MMA_BARE");

/// Raw `v_mfma_f32_32x32x8f16` emit strategy (see RocmMfma32Support). Fragments are per-lane register
/// vectors; lane l = threadIdx.x % 64 holds:
///   A (32x8):  row l%32,           k 4*(l/32) .. +3
///   B (8x32):  col l%32,           k 4*(l/32) .. +3
///   C (32x32): element e at row 8*(e/4) + 4*(l/32) + e%4, col l%32
class RocmMmaBareStrategy : public GpuMmaDispatcherStrategy {
public:
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

private:
    /// Move a fragment between memory and its lane registers (@p load: memory -> fragment).
    void emit_fragment_transfer(
        codegen::CodegenOutput& out,
        bool load,
        MmaFragmentType type,
        const GpuMmaFromMemoryLayout& layout,
        types::PrimitiveType element_type,
        const std::string& frag,
        const std::string& base
    ) const;
};

} // namespace sdfg::gpu::rocm
