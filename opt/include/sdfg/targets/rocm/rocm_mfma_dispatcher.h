#pragma once

#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

/// Raw `v_mfma_f32_32x32x8f16` dispatchers (see RocmMfma32Support). Fragments are per-lane register
/// vectors; lane l = threadIdx.x % 64 holds:
///   A (32x8):  row l%32,           k 4*(l/32) .. +3
///   B (8x32):  col l%32,           k 4*(l/32) .. +3
///   C (32x32): element e at row 8*(e/4) + 4*(l/32) + e%4, col l%32
class RocmMfmaBaseDispatcher : public RocmMmaBaseDispatcher {
public:
    using RocmMmaBaseDispatcher::RocmMmaBaseDispatcher;

protected:
    void emit_frag_zero_init(
        codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type
    ) const override;

    void emit_mma_compute(
        codegen::CodegenOutput& out,
        const std::string& frag_d_out,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c_in
    ) const override;

    void emit_eltwise_compute(
        codegen::CodegenOutput& out,
        const std::string& main_frag,
        const std::vector<std::string>& additional_frags,
        const std::function<void(
            codegen::CodegenOutput&,
            const std::string& main_frag_elem,
            const std::string& idx,
            const std::vector<std::string>& other_frag_elems
        )>& compute
    ) const override;

    void emit_needed_declarations(codegen::CodegenOutput& out) const override;

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

class RocmMfmaMatmulDispatcher : public RocmMfmaBaseDispatcher {
public:
    using RocmMfmaBaseDispatcher::RocmMfmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_single_mma(out, inputs);
    }
};

class RocmMfmaFillDispatcher : public RocmMfmaBaseDispatcher {
public:
    using RocmMfmaBaseDispatcher::RocmMfmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_accumulator_fill(out, inputs);
    }
};

class RocmMfmaFragmentLoadDispatcher : public RocmMfmaBaseDispatcher {
public:
    using RocmMfmaBaseDispatcher::RocmMfmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class RocmMfmaFragmentStoreDispatcher : public RocmMfmaBaseDispatcher {
public:
    using RocmMfmaBaseDispatcher::RocmMfmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class RocmMfmaEltwiseAddDispatcher : public RocmMfmaBaseDispatcher {
public:
    using RocmMfmaBaseDispatcher::RocmMfmaBaseDispatcher;

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override {
        dispatch_eltwise_add(out, inputs);
    }
};

} // namespace sdfg::gpu::rocm
