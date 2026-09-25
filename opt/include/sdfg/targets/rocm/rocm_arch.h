#pragma once

#include <string>
#include <vector>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_arch.h"

namespace sdfg::gpu::rocm {

struct RocmMmaSupport : public GpuMmaSupport {
    const bool f32_support;

    RocmMmaSupport(uint16_t base_size, bool f32_support, uint16_t threads)
        : GpuMmaSupport(base_size, base_size, base_size, threads), f32_support(f32_support) {}

public:
    bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const override;
    bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const override;

    std::optional<data_flow::ImplementationType> get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling)
        const override;

    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;
};


class RocmArch : public GpuArch {
    int per_cu_threads_;
    std::string rocm_name_;
    RocmMmaSupport mma_support_;

public:
    RocmArch(const std::string& name, int per_cu_threads, bool mma_base_support, bool mma_f32_support)
        : GpuArch(), rocm_name_(name), per_cu_threads_(per_cu_threads),
          mma_support_(mma_base_support ? 16 : 0, mma_f32_support, per_cu_threads) {}

    std::string unique_id() const override;
    std::string name() const override;

    int per_cu_threads() const override { return per_cu_threads_; }

    const RocmMmaSupport* mma_support() const override {
        if (mma_support_.mma_block_m > 0) {
            return &mma_support_;
        } else {
            return nullptr;
        }
    }

    structured_control_flow::ScheduleType create_schedule_type() const override;
};

extern RocmArch ROCM_ARCH_GFX1201;

extern RocmArch ROCM_ARCH_GFX90A;
extern RocmArch ROCM_ARCH_GFX942;

/// A ROCm/HIP GPU device as reported by `rocminfo`: its LLVM/ROCm ISA (gfx) name
/// together with the marketing name(s) of the device(s) that report it and the
/// wavefront size. This is the AMD analogue of @ref cuda::CudaComputeCapability.
struct RocmDeviceInfo {
    /// LLVM/ROCm ISA target name, e.g. "gfx1201", "gfx942".
    std::string gfx_name;
    /// Distinct marketing device names sharing this gfx target, useful for logging.
    std::vector<std::string> device_names;
    /// Wavefront (wave) size in threads, e.g. 32 (RDNA) or 64 (CDNA/GCN); 0 if unknown.
    int wavefront_size = 0;
};

/// Queries the ROCm GPUs available on this machine via `rocminfo`.
///
/// Internally this invokes `rocminfo` and parses its per-agent report, keeping
/// only GPU agents and matching their `gfx…` ISA name. The list is uniqued by
/// gfx name (marketing names sharing a target are collected together).
///
/// The result is cached on first call, so `rocminfo` is spawned at most once per
/// process. Returns an empty list if no ROCm device could be queried (e.g.
/// `rocminfo` is not available).
const std::vector<RocmDeviceInfo>& query_rocm_devices();

/// Resolves a queried device to a @ref RocmArch. Prefers a curated arch (with
/// correct MMA capabilities) when the gfx target is recognised; otherwise
/// synthesises a custom arch from the fields `rocminfo` reports (name +
/// wavefront, MMA disabled). The returned pointer has static lifetime.
const RocmArch* rocm_arch_for_device(const RocmDeviceInfo& device);

/// Wrapper, that allows us to store this inside the ScheduleType in the future. For now, we look at the env-vars or
/// query the local system
const RocmArch* rocm_arch_from_schedule_type(const structured_control_flow::ScheduleType& schedule);
/// Use the DOCC_ROCM_ARCH env var or if unset, query the available cards
const RocmArch* rocm_arch_from_env();
const RocmArch* rocm_arch_from_available_hardware();
const RocmArch* rocm_arch_parse(const std::string& name);

} // namespace sdfg::gpu::rocm
