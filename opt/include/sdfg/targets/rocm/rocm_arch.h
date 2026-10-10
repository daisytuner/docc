#pragma once

#include <string>
#include <vector>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/types/array.h"

namespace sdfg::gpu::rocm {

struct RocmMmaSupport : public GpuMmaSupport {
    const bool base_support;
    const bool f32_support;
    const uint16_t threads_per_mma_block;

    RocmMmaSupport(bool base_support, bool f32_support, uint16_t threads)
        : GpuMmaSupport(), base_support(base_support), f32_support(f32_support), threads_per_mma_block(threads) {
    }

protected:
    static constexpr MmaBlockSize DEFAULT_BLOCK_SIZE = {16, 16, 16};
    static constexpr MmaBlockSize CDNA_BLOCK_SIZE = {32, 32, 8};

public:
    static constexpr const char* MMA_STORAGE_TYPE = "ROCM_MMA";

    bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const override;

    types::PrimitiveType get_accumulator_type(
        types::PrimitiveType input_type, types::PrimitiveType output_type, types::PrimitiveType desired_acc_type
    ) const override;

    bool is_valid_block_size(
        const MmaBlockSize& block_size, types::PrimitiveType input_type, types::PrimitiveType acc_type
    ) const override;

    std::optional<GpuMmaTiling> try_get_mma_tiling(
        const MmaBlockSize& block_size, const symbolic::MultiExpression& res_shape, types::PrimitiveType acc_type
    ) const;

    std::optional<GpuMmaTiling> get_mma_tiling(
        const symbolic::MultiExpression& res_shape,
        types::PrimitiveType input_type,
        types::PrimitiveType acc_type,
        const MmaBlockSize* block_size_hint
    ) const override;

    void set_mma_fragment_storage_type(
        types::StorageType& storage_type, const MmaBlockSize& size, MmaFragmentType type, MmaFragmentLayout layout
    ) const override;

    static bool is_mma_type(const types::StorageType& storage);

    data_flow::ImplementationType get_mma_impl_type() const override;

    void emit_block_frag_type(
        std::ostream& os, const types::StorageType& storage_type, types::PrimitiveType element_type
    ) const;

    static void emit_block_frag_type(
        std::ostream& os,
        MmaFragmentType type,
        std::array<int, 3> dims,
        MmaFragmentLayout layout,
        types::PrimitiveType scalar_type,
        std::optional<std::pair<int, int>> coop_dims
    );

    std::vector<MmaBlockSize>
    get_supported_block_sizes(types::PrimitiveType input_type, types::PrimitiveType acc_type) const override;
};


class RocmArch : public GpuArch {
    int per_cu_threads_;
    std::string rocm_name_;
    RocmMmaSupport mma_support_;

public:
    RocmArch(const std::string& name, int per_cu_threads, bool mma_base_support, bool mma_f32_support)
        : GpuArch(), rocm_name_(name), per_cu_threads_(per_cu_threads),
          mma_support_(mma_base_support ? 16 : 0, mma_f32_support, per_cu_threads) {
    }

    std::string unique_id() const override;
    std::string name() const override;

    int per_cu_threads() const override {
        return per_cu_threads_;
    }

    const RocmMmaSupport* mma_support() const override {
        if (mma_support_.base_support) {
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
