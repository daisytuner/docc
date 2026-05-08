#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

#include "blas/dot.h"
#include "blas/gemm.h"
#include "docc/target/tenstorrent/schedule.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_node.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/plugins/plugins.h"
#include "tenstorrent_create_device.h"

namespace sdfg {
namespace tenstorrent {

inline data_flow::ImplementationType ImplementationType_Tenstorrent_WithTransfers{"TENSTORRENT_WithTransfers"};
inline data_flow::ImplementationType ImplementationType_Tenstorrent_WithoutTransfers{"TENSTORRENT_WithoutTransfers"};

/**
 * Specifically the code that will be mapped onto cores on the Tenstorrent device.
 * Expected inside a Kernel scope, which models iterating over the tiles, which will be distributed to the devices /
 * cores and will not exist as regular code in the end.
 */
class ScheduleType_Tenstorrent_Device {
public:
    static constexpr const char *BLOCKING_KEY = "blocking";
    static const std::string value() { return "TENSTORRENT_Dev"; }

    static void set_blocking(structured_control_flow::ScheduleType &schedule, bool blocking) {
        if (blocking) {
            schedule.set_property(BLOCKING_KEY, "true");
        } else {
            schedule.set_property(BLOCKING_KEY, "false");
        }
    }

    static bool is_blocking(const structured_control_flow::ScheduleType &schedule) {
        auto &props = schedule.properties();
        auto it = props.find(BLOCKING_KEY);
        if (it != props.end()) {
            return it->second == "true";
        } else {
            return false;
        }
    }

    static structured_control_flow::ScheduleType create() {
        return {value(), structured_control_flow::ScheduleTypeCategory::Offloader};
    }
};

/**
 * Offloaded in general (splits into host-support parts, abstract modeling of the streaming and the
 * device-sized-portion)
 */
class ScheduleType_Tenstorrent_Kernel {
public:
    static const std::string value() { return "TENSTORRENT"; }
    static structured_control_flow::ScheduleType create() {
        return {value(), structured_control_flow::ScheduleTypeCategory::Offloader};
    }
};

inline bool is_tenstorrent_schedule(const structured_control_flow::ScheduleType &type) {
    return type.value() == ScheduleType_Tenstorrent_Kernel::value() ||
           type.value() == ScheduleType_Tenstorrent_Device::value();
}

inline codegen::TargetType TargetType_Tenstorrent{"TENSTORRENT"};

class TenstorrentRuntimeDependency : public codegen::LibDependency {
private:
    TenstorrentRuntimeDependency() = default;

public:
    static const TenstorrentRuntimeDependency *instance() {
        static TenstorrentRuntimeDependency inst;
        return &inst;
    }

    std::string_view name() const override { return "tenstorrent_runtime"; }
    void enumerate_includes(std::vector<std::string> &out_list) const override {
        out_list.push_back("memory");
        out_list.push_back("vector");
        out_list.push_back("tt-metalium/host_api.hpp");
        out_list.push_back("tt-metalium/work_split.hpp");
        out_list.push_back("tt-metalium/tensor_accessor_args.hpp");
        out_list.push_back("daisy_rtl/global_tenstorrent_init.h");
        out_list.push_back("tracy/Tracy.hpp");
        out_list.push_back("tt-metalium/tt_metal_profiler.hpp");
    }
    std::vector<std::string_view> &globally_unique_ids() const override {
        static std::vector<std::string_view> ids{"bfloat16"};
        return ids;
    }
};

extern bool tt_emit_full_metrics;
extern bool tt_force_close_devices_after_kernel;

void register_tenstorrent_plugin(plugins::Context &docc_context);

/**
 * @deprecated, not everything can be registers w/o context
 */
void register_tenstorrent_plugin(bool emit_full_metrics = false, bool force_close_devices = false);

} // namespace tenstorrent
} // namespace sdfg
