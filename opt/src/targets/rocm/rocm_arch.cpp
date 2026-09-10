#include "sdfg/targets/rocm/rocm_arch.h"

namespace sdfg::gpu::rocm {

RocmArch ROCM_ARCH_GFX1201 = RocmArch("gfx1201", 32, true, false);

RocmArch ROCM_ARCH_GFX90A = RocmArch("gfx90a", 64, true, true);
RocmArch ROCM_ARCH_GFX942 = RocmArch("gfx942", 64, true, true);

const RocmArch* rocm_arch_parse(const std::string& name) {
    if (name == ROCM_ARCH_GFX1201.name()) {
        return &ROCM_ARCH_GFX1201;
    } else if (name == ROCM_ARCH_GFX90A.name()) {
        return &ROCM_ARCH_GFX90A;
    } else if (name == ROCM_ARCH_GFX942.name()) {
        return &ROCM_ARCH_GFX942;
    } else {
        return nullptr;
    }
}

bool RocmMmaSupport::valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const {
    // M: a rows
    // N: b cols
    // K: a cols = b rows, irrelevant to valid, as long as multiple of block size

    if (block_base == 16) {
        if (m_blocks == 1 && n_blocks == 1) {
            return true;
        } else if (m_blocks <= 2 && n_blocks <= 2) {
            return true;
        } else if (m_blocks == 8 && n_blocks == 4) { // this is limited by shared memory size, but this is the
                                                     // perf-recommend form for RDNA3
            return true;
        }
    } else if (block_base == 32) {
        if (m_blocks == 1 && n_blocks == 1) {
            return true;
        } else if (m_blocks <= 2 && n_blocks <= 2) {
            return true;
        } else if (m_blocks == 4 && n_blocks == 4) { // this is limited by shared memory size, but this is the
                                                     // perf-recommended form for CDA
            return true;
        }
    }
    return false;
}

} // namespace sdfg::gpu::rocm
