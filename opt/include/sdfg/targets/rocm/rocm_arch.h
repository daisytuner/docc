#pragma once

#include "sdfg/targets/gpu/gpu_arch.h"

namespace sdfg::gpu::rocm {

struct RocmMmaSupport : public GpuMmaSupport {
    const bool f32_support;

    RocmMmaSupport(uint16_t base_size, bool f32_support)
        : GpuMmaSupport(base_size, base_size, base_size), f32_support(f32_support) {}

public:
    bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const override;
};


class RocmArch : public GpuArch {
    int per_cu_threads_;
    RocmMmaSupport mma_support_;

public:
    RocmArch(const std::string& name, int per_cu_threads, bool mma_base_support, bool mma_f32_support)
        : GpuArch(name), per_cu_threads_(per_cu_threads), mma_support_(mma_base_support ? 16 : 0, mma_f32_support) {}

    int per_cu_threads() const override { return per_cu_threads_; }

    const RocmMmaSupport* mma_support() const override {
        if (mma_support_.mma_block_m > 0) {
            return &mma_support_;
        } else {
            return nullptr;
        }
    }
};

extern RocmArch ROCM_ARCH_GFX1201;

extern RocmArch ROCM_ARCH_GFX90A;
extern RocmArch ROCM_ARCH_GFX942;

const RocmArch* rocm_arch_parse(const std::string& name);

} // namespace sdfg::gpu::rocm
