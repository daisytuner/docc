#pragma once

#include <stdint.h>
#include <string>
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::gpu {

struct GpuMmaSupport {
    const uint16_t mma_block_m;
    const uint16_t mma_block_n;
    const uint16_t mma_block_k;

    GpuMmaSupport(uint16_t block_m, uint16_t block_n, uint16_t block_k)
        : mma_block_m(block_m), mma_block_n(block_n), mma_block_k(block_k) {}
    virtual bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const = 0;

    static int get_integer_block_count(const symbolic::Expression& size, uint16_t block_size);
};

class GpuArch {
protected:
    std::string name_;

public:
    GpuArch(const std::string& name) : name_(name) {}
    virtual ~GpuArch() = default;

    const std::string& name() const { return name_; }
    virtual int per_cu_threads() const = 0;

    virtual const GpuMmaSupport* mma_support() const = 0;
};

} // namespace sdfg::gpu
