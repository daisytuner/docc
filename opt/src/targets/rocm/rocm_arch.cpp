#include "sdfg/targets/rocm/rocm_arch.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_mma_bare_dispatcher.h"
#include "sdfg/targets/rocm/rocm_mma_fragment_dispatcher.h"

namespace sdfg::gpu::rocm {

// Int8 block K from rocWMMA's native instructions: WMMA i32_16x16x16_iu8 (RDNA4), MFMA i32_16x16x16i8 (CDNA2) and
// i32_16x16x32_i8 (CDNA3).
RocmArch ROCM_ARCH_GFX1201 = RocmArch("gfx1201", 32, true, false, 16);

RocmArch ROCM_ARCH_GFX90A = RocmArch("gfx90a", 64, true, true, 16);
RocmArch ROCM_ARCH_GFX942 = RocmArch("gfx942", 64, true, true, 32);

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

std::string rocminfo_command() {
    if (const char* rocm_env = std::getenv("ROCM_PATH"); rocm_env != nullptr && *rocm_env != '\0') {
        std::error_code ec;
        const std::filesystem::path rocminfo = std::filesystem::path(rocm_env) / "bin" / "rocminfo";
        if (std::filesystem::exists(rocminfo, ec)) {
            return rocminfo.string() + " 2>/dev/null";
        }
    }
    return "rocminfo 2>/dev/null";
}

std::vector<RocmDeviceInfo> query_rocm_devices_impl() {
    const auto output = gpu::util::run_and_capture(rocminfo_command());
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
    if (input_type == types::PrimitiveType::Int8) {
        // Int32 accumulates exactly; a narrower output would wrap.
        return int8_block_k != 0 && output_type == types::PrimitiveType::Int32;
    }
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
        // while on CDNA there are library functions that can output fp16 for fp16 inputs and bf16 for bf16 inputs,
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
    const MmaBlockSize& block_size,
    const BlockCounts* wave_tile_hint,
    const symbolic::MultiExpression& res_shape,
    types::PrimitiveType acc_type,
    bool ceil_counts
) const {
    GpuMmaTiling tiling;
    tiling.mma_block_size = block_size;
    tiling.threads_per_mma_wavefront = threads_per_mma_block;
    tiling.acc_type = acc_type;
    tiling.wave_tiling = {.blocks_m = 1, .blocks_n = 1};

    auto block_count = ceil_counts ? &GpuMmaSupport::get_ceil_block_count : &GpuMmaSupport::get_integer_block_count;
    auto mma_blocks_m = block_count(res_shape.at(0), tiling.mma_block_size.m);
    auto mma_blocks_n = block_count(res_shape.at(1), tiling.mma_block_size.n);
    auto mma_blocks_k = block_count(res_shape.at(2), tiling.mma_block_size.k);

    if (!mma_blocks_m || !mma_blocks_n || !mma_blocks_k) {
        return std::nullopt;
    }

    if (mma_blocks_m * mma_blocks_n <= 32) {
        tiling.macro_tiling = {.blocks_m = mma_blocks_m, .blocks_n = mma_blocks_n};
        // TODO: update to include wave_tiling and macro_tiling per heuristic.
        // tiling.wave_tile_blocks_m = mma_blocks_m / 2;
        // tiling.wave_tile_blocks_n = mma_blocks_n / 2;
        // tiling.macro_blocks_m = 2;
        // tiling.macro_blocks_n = 2;
    } else {
        return std::nullopt;
    }

    if (wave_tile_hint) {
        if (GpuMmaTiling::wave_tile_divides(*wave_tile_hint, tiling.macro_tiling)) {
            int waves = (tiling.macro_tiling.blocks_m / wave_tile_hint->blocks_m) *
                        (tiling.macro_tiling.blocks_n / wave_tile_hint->blocks_n);
            if (waves * tiling.threads_per_mma_wavefront <= 1024) {
                tiling.wave_tiling = *wave_tile_hint;
            } else {
                DEBUG_PRINTLN(
                    "Wave tiling hint could not be applied because it would exceed the maximum number of threads per "
                    "block (1024)."
                );
                return std::nullopt;
            }
        } else {
            DEBUG_PRINTLN("Wave tiling hint could not be applied as it does not divide the macro tiles.");
            return std::nullopt;
        }
    }

    return tiling;
}

std::optional<GpuMmaTiling> RocmMmaSupport::get_mma_tiling(
    const symbolic::MultiExpression& res_shape,
    types::PrimitiveType input_type,
    types::PrimitiveType acc_type,
    const MmaBlockSize* block_size_hint,
    const BlockCounts* wave_tile_hint
) const {
    std::optional<MmaBlockSize> desired_block_size;
    std::optional<GpuMmaTiling> mma_tiling;

    if (block_size_hint) {
        if (!is_valid_block_size(*block_size_hint, input_type, acc_type)) {
            return std::nullopt;
        }
        return try_get_mma_tiling(*block_size_hint, wave_tile_hint, res_shape, acc_type);
    } else if (input_type == types::PrimitiveType::Int8) {
        // The full-rate block first; 16x16x16 still runs if K only divides by 16.
        if (auto full_rate = try_get_mma_tiling(int8_block_size(), wave_tile_hint, res_shape, acc_type)) {
            return full_rate;
        }
        return try_get_mma_tiling(DEFAULT_BLOCK_SIZE, wave_tile_hint, res_shape, acc_type);
    } else {
        auto default_valid = try_get_mma_tiling(DEFAULT_BLOCK_SIZE, wave_tile_hint, res_shape, acc_type);
        if (default_valid.has_value()) {
            return default_valid;
        } else if (f32_support) {
            return try_get_mma_tiling(CDNA_BLOCK_SIZE, wave_tile_hint, res_shape, acc_type);
        } else {
            return std::nullopt;
        }
    }
}

std::optional<GpuMmaTiling> RocmMmaSupport::get_ragged_mma_tiling(
    const symbolic::MultiExpression& padded_shape,
    types::PrimitiveType input_type,
    types::PrimitiveType output_type,
    types::PrimitiveType acc_type,
    const MmaBlockSize* block_size_hint,
    const BlockCounts* wave_tile_hint
) const {
    // Masked fragment transfers recover element coordinates from an iota fragment of the element type, which
    // must represent every index of a block exactly (16-bit floats: all integers up to 2048).
    auto exact_iota = [](types::PrimitiveType t) {
        return t == types::PrimitiveType::Half || t == types::PrimitiveType::Float;
    };
    if (!exact_iota(input_type) || !exact_iota(output_type) || !exact_iota(acc_type)) {
        return std::nullopt;
    }
    if (block_size_hint) {
        if (!is_valid_block_size(*block_size_hint, input_type, acc_type)) {
            return std::nullopt;
        }
        return try_get_mma_tiling(*block_size_hint, wave_tile_hint, padded_shape, acc_type, true);
    }
    for (const auto& block : {DEFAULT_BLOCK_SIZE, CDNA_BLOCK_SIZE}) {
        if (is_valid_block_size(block, input_type, acc_type)) {
            if (auto tiling = try_get_mma_tiling(block, wave_tile_hint, padded_shape, acc_type, true)) {
                return tiling;
            }
        }
    }
    return std::nullopt;
}

void RocmMmaSupport::set_mma_fragment_storage_type(
    types::StorageType& storage_type, const GpuMmaTiling& tiling, MmaFragmentType type, MmaFragmentLayout layout
) const {
    auto variant = select_variant(tiling);
    if (variant == RocmMmaVariant::Bare) {
        storage_type.value(MMA_REGS_STORAGE_TYPE);
    } else if (variant == RocmMmaVariant::Fragment) {
        storage_type.value(MMA_FRAGMENT_STORAGE_TYPE);
    } else {
        throw std::invalid_argument("Unknown RocmMmaVariant: " + std::to_string(static_cast<int>(variant)));
    }
    auto& size = tiling.mma_block_size;
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
    if (input_type == types::PrimitiveType::Int8) {
        return int8_block_k != 0 && acc_type == types::PrimitiveType::Int32 && block_size.m == 16 &&
               block_size.n == 16 && (block_size.k == 16 || block_size.k == int8_block_k);
    }
    if (block_size.m == 16 && block_size.n == 16 && block_size.k == 16) {
        return true;
    } else if (block_size.m == 32 && block_size.n == 32 && block_size.k == 8) {
        // 32x32x8 is a CDNA-only (MFMA) block size; RDNA (e.g. gfx1201) lacks it. For now only supporting fp16->fp32
        // accumulation.
        return f32_support && input_type == types::PrimitiveType::Half && acc_type == types::PrimitiveType::Float;
    } else {
        return false;
    }
}

bool RocmMmaSupport::is_mma_type(const types::StorageType& storage) {
    return storage.value() == MMA_FRAGMENT_STORAGE_TYPE || storage.value() == MMA_REGS_STORAGE_TYPE;
}

RocmMmaVariant RocmMmaSupport::select_variant(const GpuMmaTiling& tiling) {
    auto& block_size = tiling.mma_block_size;
    const bool is_32_block = block_size.m == 32 && block_size.n == 32 && block_size.k == 8;
    if (tiling.impl_details) {
        if (static_cast<RocmMmaVariant>(tiling.impl_details) == RocmMmaVariant::Bare && !is_32_block) {
            throw std::
                invalid_argument("RocmMmaBareStrategy (MFMA) only supports 32x32x8 blocks, got " + block_size.toStr());
        }
        return static_cast<RocmMmaVariant>(tiling.impl_details);
    }
    return is_32_block ? RocmMmaVariant::Bare : RocmMmaVariant::Fragment;
}

data_flow::ImplementationType RocmMmaSupport::get_mma_impl_type(const GpuMmaTiling& tiling) const {
    auto variant = select_variant(tiling);

    if (variant == RocmMmaVariant::Bare) {
        return ImplementationType_ROCM_MMA_BARE;
    } else if (variant == RocmMmaVariant::Fragment) {
        return ImplementationType_ROCM_MMA_FRAGMENT;
    } else {
        throw std::invalid_argument("Unknown RocmMmaVariant: " + std::to_string(static_cast<int>(variant)));
    }
}

void RocmMmaSupport::emit_block_frag_type(
    std::ostream& os, const types::StorageType& storage_type, types::PrimitiveType element_type
) const {
    MmaBlockSize size{
        get_storage_type_arg_as_int(storage_type, 0),
        get_storage_type_arg_as_int(storage_type, 1),
        get_storage_type_arg_as_int(storage_type, 2)
    };
    auto type = static_cast<MmaFragmentType>(get_storage_type_arg_as_int(storage_type, 3));
    auto layout = static_cast<MmaFragmentLayout>(get_storage_type_arg_as_int(storage_type, 4));
    auto val = storage_type.value();
    if (val == MMA_FRAGMENT_STORAGE_TYPE) {
        emit_block_frag_type(os, type, size, layout, element_type, std::nullopt);
    } else if (val == MMA_REGS_STORAGE_TYPE) {
        emit_bare_fragment_decl_type(os, type, size, layout, element_type, std::nullopt);
    }
}

void RocmMmaSupport::emit_bare_fragment_decl_type(
    std::ostream& os,
    MmaFragmentType type,
    const MmaBlockSize& dims,
    MmaFragmentLayout layout,
    types::PrimitiveType scalar_type,
    std::optional<std::pair<int, int>> coop_dims
) {
    auto shape = dims.get_shape(type);
    const auto elems =
        SymEngine::rcp_static_cast<const SymEngine::Integer>(symbolic::mul(shape.at(0), shape.at(1)))->as_int() / 64;
    switch (scalar_type) {
        case types::PrimitiveType::Half:
            os << "_Float16";
            break;
        case types::PrimitiveType::Float:
            os << "float";
            break;
        default:
            throw std::invalid_argument(
                "invalid scalar type: " + std::string(types::primitive_type_to_string(scalar_type)) +
                " on mfma fragment declaration"
            );
    }
    os << " __attribute__((ext_vector_type(" << elems << ")))";
}

void RocmMmaSupport::emit_block_frag_type(
    std::ostream& os,
    MmaFragmentType type,
    const MmaBlockSize& dims,
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
    os << dims.m << ", ";
    os << dims.n << ", ";
    os << dims.k << ", ";
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
        case types::PrimitiveType::Int8:
            os << "int8_t";
            break;
        case types::PrimitiveType::Int32:
            os << "int32_t";
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
    if (input_type == types::PrimitiveType::Int8) {
        if (int8_block_k != 0 && acc_type == types::PrimitiveType::Int32) {
            supported_sizes.push_back(int8_block_size());
            if (int8_block_k != DEFAULT_BLOCK_SIZE.k) {
                supported_sizes.push_back(DEFAULT_BLOCK_SIZE);
            }
        }
        return supported_sizes;
    }
    if (types::is_floating_point(acc_type)) {
        supported_sizes.push_back(DEFAULT_BLOCK_SIZE);
    }
    if (f32_support) {
        supported_sizes.push_back(CDNA_BLOCK_SIZE);
    }
    return supported_sizes;
}

} // namespace sdfg::gpu::rocm
