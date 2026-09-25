#include "sdfg/targets/rocm/rocm_arch.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

RocmArch ROCM_ARCH_GFX1201 = RocmArch("gfx1201", 32, true, false);

RocmArch ROCM_ARCH_GFX90A = RocmArch("gfx90a", 64, true, true);
RocmArch ROCM_ARCH_GFX942 = RocmArch("gfx942", 64, true, true);

const RocmArch* rocm_arch_parse(const std::string& raw_name) {
    // Strip an optional case-insensitive "cuda:" prefix before matching.
    std::string name = raw_name;
    if (name.size() >= 5) {
        std::string prefix = name.substr(0, 5);
        std::transform(prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (prefix == "rocm:") {
            name = name.substr(5);
        }
    }
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

namespace {

/// Parses the integer prefix of a rocminfo value such as "32(0x20)" -> 32.
/// Returns 0 when no leading digits are present.
int parse_leading_int(const std::string& value) {
    std::size_t i = 0;
    while (i < value.size() && std::isdigit(static_cast<unsigned char>(value[i])) != 0) {
        ++i;
    }
    if (i == 0) {
        return 0;
    }
    try {
        return std::stoi(value.substr(0, i));
    } catch (const std::exception&) {
        return 0;
    }
}

/// Parses a full `rocminfo` report into the list of GPU devices.
///
/// `rocminfo` prints one block per HSA agent (both CPU and GPU). Within a GPU
/// agent, the clean ISA name (e.g. "gfx942") and the wavefront size appear at
/// the agent level first, then again — with target-feature suffixes — inside the
/// nested `ISA Info` block. We therefore keep the *first* occurrence per agent.
std::vector<RocmDeviceInfo> parse_rocminfo(const std::string& report) {
    struct AgentAcc {
        std::string name;
        std::string marketing;
        std::string device_type;
        int wavefront = 0;
        bool active = false;
    };

    std::map<std::string, RocmDeviceInfo> devices; // keyed + ordered by gfx name
    AgentAcc cur;

    auto flush = [&]() {
        if (cur.active && cur.device_type == "GPU" && cur.name.rfind("gfx", 0) == 0) {
            auto& info = devices[cur.name];
            info.gfx_name = cur.name;
            if (info.wavefront_size == 0) {
                info.wavefront_size = cur.wavefront;
            }
            const std::string dev = cur.marketing.empty() ? cur.name : cur.marketing;
            if (std::find(info.device_names.begin(), info.device_names.end(), dev) == info.device_names.end()) {
                info.device_names.push_back(dev);
            }
        }
        cur = AgentAcc{};
    };

    std::istringstream lines(report);
    std::string line;
    while (std::getline(lines, line)) {
        const std::string trimmed = gpu::util::trim(line);
        if (trimmed.rfind("Agent", 0) == 0) {
            flush();
            cur.active = true;
            continue;
        }
        const auto colon = trimmed.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = gpu::util::trim(trimmed.substr(0, colon));
        const std::string value = gpu::util::trim(trimmed.substr(colon + 1));
        if (key == "Name") {
            // Agent-level name comes before the ISA-level one; keep the first.
            if (cur.name.empty()) {
                cur.name = value;
            }
        } else if (key == "Marketing Name") {
            cur.marketing = value;
        } else if (key == "Device Type") {
            cur.device_type = value;
        } else if (key == "Wavefront Size") {
            if (cur.wavefront == 0) {
                cur.wavefront = parse_leading_int(value);
            }
        }
    }
    flush();

    std::vector<RocmDeviceInfo> result;
    result.reserve(devices.size());
    for (auto& [gfx, info] : devices) {
        result.push_back(std::move(info));
    }
    return result;
}

std::vector<RocmDeviceInfo> query_rocm_devices_impl() {
    const auto output = gpu::util::run_and_capture("rocminfo 2>/dev/null");
    if (!output) {
        return {};
    }
    return parse_rocminfo(*output);
}

} // namespace

const std::vector<RocmDeviceInfo>& query_rocm_devices() {
    // Cache the parsed report so `rocminfo` is spawned at most once per process.
    static const std::vector<RocmDeviceInfo> cached = query_rocm_devices_impl();
    return cached;
}

const RocmArch* rocm_arch_for_device(const RocmDeviceInfo& device) {
    // Prefer a curated arch (correct MMA capabilities) when we recognise the gfx target.
    if (const RocmArch* known = rocm_arch_parse(device.gfx_name)) {
        return known;
    }
    // Unknown target: synthesise a RocmArch from what rocminfo reports (name +
    // wavefront). MMA capabilities aren't in the report, so disable them. The
    // map is node-based, so the returned pointer stays valid for the process.
    static std::map<std::string, RocmArch> custom;
    auto it = custom.find(device.gfx_name);
    if (it == custom.end()) {
        const int wavefront = device.wavefront_size > 0 ? device.wavefront_size : 64;
        it = custom.emplace(device.gfx_name, RocmArch(device.gfx_name, wavefront, false, false)).first;
    }
    return &it->second;
}

const RocmArch* rocm_arch_from_schedule_type(const structured_control_flow::ScheduleType& schedule) {
    auto it = schedule.properties().find(GpuArch::ARCH_PROPERTY);
    if (it != schedule.properties().end()) {
        return rocm_arch_parse(it->second);
    } else if (auto from_env = rocm_arch_from_env()) {
        return from_env;
    } else {
        return rocm_arch_from_available_hardware();
    }
}

std::string RocmArch::unique_id() const { return "Rocm:" + rocm_name_; }

std::string RocmArch::name() const { return rocm_name_; }

structured_control_flow::ScheduleType RocmArch::create_schedule_type() const {
    structured_control_flow::ScheduleType
        sched(sdfg::rocm::ScheduleType_ROCM_Offload::value(), structured_control_flow::ScheduleTypeCategory::Offloader);
    sched.set_property(ARCH_PROPERTY, unique_id());
    return sched;
}

const RocmArch* rocm_arch_from_env() {
    const char* env = std::getenv("DOCC_ROCM_ARCH");
    if (env && !std::string(env).empty()) {
        auto* arch = rocm_arch_parse(env);
        if (!arch) {
            throw std::runtime_error(std::string("Unknown ROCm architecture in DOCC_ROCM_ARCH: ") + env);
        }
        return arch;
    } else {
        return nullptr;
    }
}

const RocmArch* rocm_arch_from_available_hardware() {
    const auto& devices = query_rocm_devices();
    if (devices.empty()) {
        return nullptr;
    }
    return rocm_arch_for_device(devices.front());
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
        } else if (m_blocks == 4 && n_blocks == 4) {
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

bool RocmMmaSupport::supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const {
    if (input_type == types::PrimitiveType::BFloat || input_type == types::PrimitiveType::Half) {
        return output_type == types::PrimitiveType::Float || output_type == input_type;
    } else if (input_type == types::PrimitiveType::Float) {
        return f32_support && output_type == types::PrimitiveType::Float;
    } else if (input_type == types::PrimitiveType::Double) {
        return f32_support && output_type == types::PrimitiveType::Double;
    }
    return false;
}

GpuMmaTiling RocmMmaSupport::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
    GpuMmaTiling tiling;
    tiling.mma_block_m = mma_block_m;
    tiling.mma_block_n = mma_block_n;
    tiling.mma_block_k = mma_block_k;
    tiling.threads_per_mma_block_m = threads_per_mma_block;

    auto mma_blocks_m = get_integer_block_count(res_shape.at(0), tiling.mma_block_m);
    auto mma_blocks_n = get_integer_block_count(res_shape.at(1), tiling.mma_block_n);
    auto mma_blocks_k = get_integer_block_count(res_shape.at(2), tiling.mma_block_k);

    if (!mma_blocks_m || !mma_blocks_n || !mma_blocks_k) {
        throw std::runtime_error("Result shape is not compatible with MMA block sizes.");
    }
    if (mma_blocks_m == 1 && mma_blocks_n == 1) {
        tiling.wave_tile_blocks_m = 1;
        tiling.wave_tile_blocks_n = 1;
        tiling.macro_blocks_m = 1;
        tiling.macro_blocks_n = 1;
    } else if (mma_blocks_m <= 2 && mma_blocks_n <= 2) {
        tiling.wave_tile_blocks_m = 1;
        tiling.wave_tile_blocks_n = 1;
        tiling.macro_blocks_m = mma_blocks_m;
        tiling.macro_blocks_n = mma_blocks_n;
    } else if (mma_blocks_n <= 4 && (mma_blocks_m == 4 || mma_blocks_m == 8)) {
        tiling.wave_tile_blocks_m = 1;
        tiling.wave_tile_blocks_n = 1;
        tiling.macro_blocks_m = mma_blocks_m;
        tiling.macro_blocks_n = mma_blocks_n;
        // tiling.mma_block_m *= 2;
        // tiling.mma_block_n += 2;
        // tiling.wave_tile_blocks_m = mma_blocks_m / 2;
        // tiling.wave_tile_blocks_n = mma_blocks_n / 2;
        // tiling.macro_blocks_m = 2;
        // tiling.macro_blocks_n = 2;
    } else {
        throw std::runtime_error("Unsupported MMA block configuration for this GPU target.");
    }

    return tiling;
}

std::optional<data_flow::ImplementationType> RocmMmaSupport::
    get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling) const {
    auto arch_name = arch.name();
    if (arch_name == "gfx1201") {
        return ImplementationType_ROCM_MMA_GFX1201;
    } else if (arch_name == "gfx90a") {
        return ImplementationType_ROCM_MMA_GFX90A;
    }
    return std::nullopt;
}

} // namespace sdfg::gpu::rocm
