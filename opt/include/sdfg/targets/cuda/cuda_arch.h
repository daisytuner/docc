#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_arch.h"

namespace sdfg::gpu::cuda {

/// A CUDA compute capability together with the name(s) of the device(s) that
/// report it.
struct CudaComputeCapability {
    /// Compute capability in clang's integer convention (e.g. 8.6 -> 86,
    /// 12.0 -> 120).
    uint32_t compute_cap;
    /// Distinct device names that share this compute capability, useful for
    /// logging.
    std::vector<std::string> device_names;
};

/// Queries the CUDA compute capabilities of the GPUs available on this machine.
///
/// Internally this invokes `nvidia-smi --query-gpu=name,compute_cap` and parses
/// its output. Each device's compute capability is reported as an integer using
/// the same convention as clang (e.g. `8.6` -> `86`, `12.0` -> `120`).
///
/// The returned list is uniqued by compute capability (the names of all devices
/// sharing a capability are collected together) and sorted from the highest to
/// the lowest compute capability.
///
/// @return A descending, duplicate-free list of the available compute
///         capabilities, or an empty vector if no CUDA device could be queried
///         (e.g. `nvidia-smi` is not available).
std::vector<CudaComputeCapability> query_cuda_compute_capabilities();

struct CudaMmaSupport : public GpuMmaSupport {
    const bool tf32_support;
    const bool fp64_support;

    CudaMmaSupport(uint16_t base_size, bool tf32_support, bool fp64_support)
        : GpuMmaSupport(base_size, base_size, base_size, 32), tf32_support(tf32_support), fp64_support(fp64_support) {}

public:
    bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const override;
    bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const override;
    std::optional<data_flow::ImplementationType> get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling)
        const override;
    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;
};


class CudaArch : public GpuArch {
    int sm_version_;
    CudaMmaSupport mma_support_;

public:
    CudaArch(int sm_version, bool mma_base_support, bool mma_tf32_support, bool mma_fp64_support)
        : GpuArch("sm_" + std::to_string(sm_version)), sm_version_(sm_version),
          mma_support_(mma_base_support ? 16 : 0, mma_tf32_support, mma_fp64_support) {}

    int per_cu_threads() const override { return 32; }

    const CudaMmaSupport* mma_support() const override {
        if (mma_support_.mma_block_m > 0) {
            return &mma_support_;
        } else {
            return nullptr;
        }
    }

    structured_control_flow::ScheduleType create_schedule_type() const override;
};

extern CudaArch CUDA_ARCH_SM70;
extern CudaArch CUDA_ARCH_SM75;
extern CudaArch CUDA_ARCH_SM80;
extern CudaArch CUDA_ARCH_SM86;
extern CudaArch CUDA_ARCH_SM87;
extern CudaArch CUDA_ARCH_SM89;
extern CudaArch CUDA_ARCH_SM90;
extern CudaArch CUDA_ARCH_SM100;
extern CudaArch CUDA_ARCH_SM103;
extern CudaArch CUDA_ARCH_SM107;
extern CudaArch CUDA_ARCH_SM110;
extern CudaArch CUDA_ARCH_SM120;

/// Wrapper, that allows us to store this inside the ScheduleType in the future. For now, we look at the env-vars or
/// query the local system
const CudaArch* cuda_arch_from_schedule_type(const structured_control_flow::ScheduleType& schedule);
const CudaArch* cuda_arch_from_env();
const CudaArch* cuda_arch_from_available_hardware();
const CudaArch* cuda_arch_parse(const std::string& name);

} // namespace sdfg::gpu::cuda
