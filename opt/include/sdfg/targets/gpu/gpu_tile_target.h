#pragma once

#include <optional>
#include <string>
#include <utility>

#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/tiles/tile_target.h"

namespace sdfg {
namespace tiles {

/**
 * @brief The built-in GPU backend for the tile algebra, shared by CUDA and ROCm.
 *
 * Maps an offload schedule's target level to a cooperation @ref Level and the
 * abstract @ref Space tiers to NVIDIA/CDNA buffers. @p lane_width is the
 * warp/wavefront size (32 on NVIDIA, 64 on CDNA); @p offload_value is the
 * `*_Offload` schedule value this instance owns (the legacy fused schedule is the
 * other value it is registered under).
 */
class GPUTileTarget : public TileTarget {
    unsigned lane_width_;
    std::string offload_value_;

    static unsigned spatial_axis_of(gpu::TargetLevel tl) {
        switch (tl) {
            case gpu::TargetLevel::Y_GRID:
            case gpu::TargetLevel::Y_BLOCK:
                return 1;
            case gpu::TargetLevel::Z_GRID:
            case gpu::TargetLevel::Z_BLOCK:
                return 2;
            default:
                return 0;
        }
    }

public:
    GPUTileTarget(unsigned lane_width, std::string offload_value)
        : lane_width_(lane_width), offload_value_(std::move(offload_value)) {}

    std::optional<AxisSchedule> classify(const structured_control_flow::ScheduleType& sched) const override {
        if (sched.value() == offload_value_) {
            gpu::TargetLevel tl = gpu::gpu_target_level(sched);
            Level level = gpu::is_grid_level(tl)    ? Level::Device
                          : gpu::is_block_level(tl) ? Level::Group
                                                    : Level::Subgroup;
            return AxisSchedule(
                level,
                space(level),
                has_scratchpad(),
                spatial_axis_of(tl),
                gpu::ScheduleType_GPU_Offload::parallel_size(sched),
                gpu::ScheduleType_GPU_Offload::nested_sync(sched)
            );
        }
        // Legacy fused block-thread schedule.
        return AxisSchedule(
            Level::Group, space(Level::Group), has_scratchpad(), /*spatial_axis=*/0, gpu::gpu_block_size(sched)
        );
    }

    /// The canonical scratchpad hierarchy: Device->Global, Group->Shared, Subgroup->Register.
    Space space(Level level) const override { return ::sdfg::tiles::default_space(level); }

    bool supports_cooperative_staging(const structured_control_flow::ScheduleType& sched) const override {
        // Only the offload schedule carries a separable block-thread copy Map; the
        // legacy fused schedule cannot host cooperative shared-memory staging.
        return sched.value() == offload_value_;
    }

    types::StorageType storage_type(Space space) const override {
        switch (space) {
            case Space::Shared:
                return types::StorageType::NV_Shared();
            case Space::Global:
                return types::StorageType::NV_Global();
            case Space::Register:
                return types::StorageType::CPU_Stack();
        }
        return types::StorageType::CPU_Stack();
    }

    unsigned lane_width() const override { return lane_width_; }
};

} // namespace tiles
} // namespace sdfg
