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

std::string RocmArch::unique_id() const {
    return "Rocm:" + rocm_name_;
}

std::string RocmArch::name() const {
    return rocm_name_;
}

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

types::PrimitiveType RocmMmaSupport::get_accumulator_type(
    types::PrimitiveType input_type, types::PrimitiveType output_type, types::PrimitiveType desired_acc_type
) const {
    // assumes input & output types are supported by this architecture, as checked by supported_types()

    if (types::is_floating_point(input_type) && types::bit_width(input_type) < 32) {
        // while on CDNA there are lib-functionsthat can output fp16 for fp16 inputs and bf16 for bf16 inputs,
        // the MMA instructions themselves always accumulate to fp32
        // and the conversion will happen while reading the data with additional slowdown. In all current usecases,
        // we can handle converting to the external output type before writeback if needed.
        if (!f32_support && desired_acc_type != types::PrimitiveType::Void) {
            return desired_acc_type;
        } else {
            return types::PrimitiveType::Float;
        }
    } else if (input_type == types::PrimitiveType::Float) {
        return types::PrimitiveType::Float;
    } else if (input_type == types::PrimitiveType::Double) {
        return types::PrimitiveType::Double;
    } else if (types::is_integer(input_type)) {
        return types::PrimitiveType::Int32;
    } else {
        throw std::runtime_error("Unsupported MMA input type: " + std::string(types::primitive_type_to_string(input_type)));
    }
}

std::optional<GpuMmaTiling> RocmMmaSupport::try_get_mma_tiling(
    const MmaBlockSize& block_size, const symbolic::MultiExpression& res_shape, types::PrimitiveType acc_type
) const {
    GpuMmaTiling tiling;
    tiling.mma_block_size = block_size;
    tiling.threads_per_mma_block_m = threads_per_mma_block;
    tiling.acc_type = acc_type;

    auto mma_blocks_m = get_integer_block_count(res_shape.at(0), tiling.mma_block_size.m);
    auto mma_blocks_n = get_integer_block_count(res_shape.at(1), tiling.mma_block_size.n);
    auto mma_blocks_k = get_integer_block_count(res_shape.at(2), tiling.mma_block_size.k);

    if (!mma_blocks_m || !mma_blocks_n || !mma_blocks_k) {
        return std::nullopt;
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
        return std::nullopt;
    }

    return tiling;
}

std::optional<GpuMmaTiling> RocmMmaSupport::get_mma_tiling(
    const symbolic::MultiExpression& res_shape,
    types::PrimitiveType input_type,
    types::PrimitiveType acc_type,
    const MmaBlockSize* block_size_hint
) const {
    std::optional<MmaBlockSize> desired_block_size;

    if (block_size_hint) {
        if (is_valid_block_size(*block_size_hint, input_type, acc_type)) {
            desired_block_size = *block_size_hint;
        }
    }

    std::optional<GpuMmaTiling> mma_tiling;
    if (desired_block_size) {
        mma_tiling = try_get_mma_tiling(desired_block_size.value(), res_shape, acc_type);
    }

    if (!mma_tiling) {
        // fallback to default block size
        mma_tiling = try_get_mma_tiling(DEFAULT_BLOCK_SIZE, res_shape, acc_type);
    }

    return mma_tiling;
}

void RocmMmaSupport::set_mma_fragment_storage_type(
    types::StorageType& storage_type, const MmaBlockSize& size, MmaFragmentType type, MmaFragmentLayout layout
) const {
    storage_type.value(MMA_STORAGE_TYPE);
    storage_type.args(
        {symbolic::integer(size.m),
         symbolic::integer(size.n),
         symbolic::integer(size.k),
         symbolic::integer(static_cast<int>(type)),
         symbolic::integer(static_cast<int>(layout))}
    );
}

bool RocmMmaSupport::is_valid_block_size(
    const MmaBlockSize& block_size, types::PrimitiveType input_type, types::PrimitiveType acc_type
) const {
    if (block_size.m == 16 && block_size.n == 16 && block_size.k == 16) {
        return true;
    } else if (block_size.m == 32 && block_size.n == 32 && block_size.k == 8) {
        // 32x32x8 is a CDNA-only (MFMA) block size; RDNA (e.g. gfx1201) lacks it.
        return f32_support;
    } else {
        return false;
    }
}

bool RocmMmaSupport::is_mma_type(const types::StorageType& storage) {
    return storage.value() == MMA_STORAGE_TYPE;
}

data_flow::ImplementationType RocmMmaSupport::get_mma_impl_type() const {
    return ImplementationType_ROCM_MMA;
}

void RocmMmaSupport::emit_block_frag_type(
    std::ostream& os, const types::StorageType& storage_type, types::PrimitiveType element_type
) const {
    emit_block_frag_type(
        os,
        static_cast<MmaFragmentType>(get_storage_type_arg_as_int(storage_type, 3)),
        {get_storage_type_arg_as_int(storage_type, 0),
         get_storage_type_arg_as_int(storage_type, 1),
         get_storage_type_arg_as_int(storage_type, 2)},
        static_cast<MmaFragmentLayout>(get_storage_type_arg_as_int(storage_type, 4)),
        element_type,
        std::nullopt
    );
}

void RocmMmaSupport::emit_block_frag_type(
    std::ostream& os,
    MmaFragmentType type,
    std::array<int, 3> dims,
    MmaFragmentLayout layout,
    types::PrimitiveType scalar_type,
    std::optional<std::pair<int, int>> coop_dims
) {
    os << "rocwmma::fragment<";
    switch (type) {
        case MmaFragmentType::A:
            os << "rocwmma::matrix_a, ";
            break;
        case MmaFragmentType::B:
            os << "rocwmma::matrix_b, ";
            break;
        case MmaFragmentType::C:
            os << "rocwmma::accumulator, ";
            break;
        default:
            throw std::invalid_argument("invalid fragment type");
    }
    os << dims[0] << ", ";
    os << dims[1] << ", ";
    os << dims[2] << ", ";
    switch (scalar_type) {
        case types::PrimitiveType::BFloat:
            os << "rocwmma::bfloat16_t";
            break;
        case types::PrimitiveType::Half:
            os << "rocwmma::float16_t";
            break;
        case types::PrimitiveType::Float:
            os << "rocwmma::float32_t";
            break;
        default:
            throw std::invalid_argument(
                "invalid scalar type: " + std::string(types::primitive_type_to_string(scalar_type)) +
                " on mma fragment declaration"
            );
    }

    if (layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
        os << ", rocwmma::col_major";
    } else if (layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
        os << ", rocwmma::row_major";
    }

    if (coop_dims) {
        os << ", rocwmma::fragment_scheduler::coop_row_major_2d<" << coop_dims->first << ", " << coop_dims->second
           << ">";
    }
    os << ">";
}

std::vector<MmaBlockSize> RocmMmaSupport::
    get_supported_block_sizes(types::PrimitiveType input_type, types::PrimitiveType acc_type) const {
    std::vector<MmaBlockSize> supported_sizes;
    if (acc_type == types::is_floating_point(acc_type)) {
        supported_sizes.push_back(DEFAULT_BLOCK_SIZE);
    }
    if (f32_support) {
        supported_sizes.push_back(CDNA_BLOCK_SIZE);
    }
    return supported_sizes;
}

} // namespace sdfg::gpu::rocm
