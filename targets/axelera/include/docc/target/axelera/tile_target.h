#pragma once

#include <sdfg/tiles/tile_target.h>

namespace sdfg {
namespace axelera {

/// AI-core-local memory: the only memory the IFDs and the ODR stream from and to.
inline types::StorageType StorageType_L1() {
    return types::StorageType("AX_L1");
}

/// Memory shared by all AI cores.
inline types::StorageType StorageType_L2() {
    return types::StorageType("AX_L2");
}

/// The in-memory-compute array of an AI core, holding the matmul weights resident.
inline types::StorageType StorageType_IMC() {
    return types::StorageType("AX_IMC");
}

inline data_flow::ImplementationType ImplementationType_Axelera{"AXELERA"};

/**
 * Makes AXELERA loops legible to the tile algebra. An AI core runs its region alone, so no AXELERA loop cooperates
 * on a tile: every tile staged inside a core is core-private and lives in its L1.
 */
class AxeleraTileTarget : public tiles::TileTarget {
public:
    std::optional<tiles::AxisSchedule> classify(const structured_control_flow::ScheduleType&) const override {
        return std::nullopt;
    }

    tiles::Space space(tiles::Level level) const override {
        switch (level) {
            case tiles::Level::Device:
                return tiles::Space::Global;
            case tiles::Level::Group:
                return tiles::Space::Shared;
            default:
                return tiles::Space::Register;
        }
    }

    bool supports_cooperative_staging(const structured_control_flow::ScheduleType&) const override {
        return false;
    }

    types::StorageType storage_type(tiles::Space space) const override {
        switch (space) {
            case tiles::Space::Register:
                return StorageType_L1();
            case tiles::Space::Shared:
                return StorageType_L2();
            default:
                return types::StorageType::CPU_Heap();
        }
    }

    unsigned lane_width() const override {
        return 1;
    }

    data_flow::ImplementationType implementation_type() const override {
        return ImplementationType_Axelera;
    }

    /// The core's DMA fills L1 tiles while its datapaths run, so L1 staging buffers can be multi-buffered.
    bool supports_async_staging(const types::StorageType& storage) const override {
        return storage == StorageType_L1();
    }
};

} // namespace axelera
} // namespace sdfg
