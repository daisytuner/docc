#include "sdfg/targets/cuda/cuda_arch.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_arch.h"

namespace sdfg::gpu::cuda {

namespace {

/// Parses a compute capability string like "8.6" into the integer form clang
/// uses (86). "12.0" becomes 120. Returns std::nullopt on malformed input.
static std::optional<uint32_t> cuda_parse_compute_cap(const std::string& raw) {
    const std::string trimmed = gpu::util::trim(raw);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    const auto dot = trimmed.find('.');
    if (dot == std::string::npos) {
        return std::nullopt;
    }

    const std::string major_str = trimmed.substr(0, dot);
    const std::string minor_str = trimmed.substr(dot + 1);
    if (major_str.empty() || minor_str.empty()) {
        return std::nullopt;
    }

    try {
        const int major = std::stoi(major_str);
        const int minor = std::stoi(minor_str);
        // clang convention: major * 10 + minor (e.g. 8.6 -> 86, 12.0 -> 120).
        return static_cast<uint32_t>(major * 10 + minor);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<CudaComputeCapability> query_cuda_compute_capabilities_impl() {
    const auto output =
        gpu::util::run_and_capture("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null");
    if (!output) {
        return {};
    }

    // Collect the distinct device names per compute capability. std::map keeps
    // the capabilities ordered so we can emit them descending below.
    std::map<uint32_t, std::vector<std::string>> caps;
    std::istringstream lines(*output);
    std::string line;
    while (std::getline(lines, line)) {
        if (gpu::util::trim(line).empty()) {
            continue;
        }
        // Output line format is "<name>, <compute_cap>"; the compute cap is the
        // last comma-separated field, the name is everything before it.
        const auto comma = line.find_last_of(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::string name = gpu::util::trim(line.substr(0, comma));
        const auto cap = cuda_parse_compute_cap(line.substr(comma + 1));
        if (!cap) {
            continue;
        }

        auto& names = caps[*cap];
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }

    // Emit highest to lowest compute capability.
    std::vector<CudaComputeCapability> result;
    result.reserve(caps.size());
    for (auto it = caps.rbegin(); it != caps.rend(); ++it) {
        result.push_back({.compute_cap = it->first, .device_names = std::move(it->second)});
    }
    return result;
}

} // namespace

std::vector<CudaComputeCapability> query_cuda_compute_capabilities() {
    // Cache the parsed result so `nvidia-smi` is spawned at most once per process.
    static const std::vector<CudaComputeCapability> cached = query_cuda_compute_capabilities_impl();
    return cached;
}

bool CudaMmaSupport::supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const {
    if (input_type == types::PrimitiveType::Half) {
        return this->mma_block_m > 0;
    } else if (input_type == types::PrimitiveType::BFloat) { // TF32 would go here
        return this->tf32_support;
    } else if (input_type == types::PrimitiveType::Double) {
        return this->fp64_support;
    } else {
        return false;
    }
}

std::optional<data_flow::ImplementationType> CudaMmaSupport::
    get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling) const {
    return std::nullopt;
}

GpuMmaTiling CudaMmaSupport::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
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

bool CudaMmaSupport::valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const {
    // very preliminary table
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

    return false;
}

const CudaArch* cuda_arch_from_schedule_type(const structured_control_flow::ScheduleType& schedule) {
    auto it = schedule.properties().find(GpuArch::ARCH_PROPERTY);
    if (it != schedule.properties().end()) {
        return cuda_arch_parse(it->second);
    } else if (auto from_env = cuda_arch_from_env()) {
        return from_env;
    } else {
        return cuda_arch_from_available_hardware();
    }
}

std::string CudaArch::unique_id() const { return "Cuda:sm_" + std::to_string(sm_version_); }

std::string CudaArch::name() const { return "sm_" + std::to_string(sm_version_); }

structured_control_flow::ScheduleType CudaArch::create_schedule_type() const {
    structured_control_flow::ScheduleType
        sched(sdfg::cuda::ScheduleType_CUDA_Offload::value(), structured_control_flow::ScheduleTypeCategory::Offloader);
    sched.set_property("ARCH", unique_id());
    return sched;
}

const CudaArch* cuda_arch_from_env() {
    const char* env = std::getenv("DOCC_CUDA_ARCH");
    if (env && !std::string(env).empty()) {
        auto* arch = cuda_arch_parse(env);
        if (!arch) {
            throw std::runtime_error(std::string("Unknown CUDA architecture in DOCC_CUDA_ARCH: ") + env);
        }
        return arch;
    } else {
        return nullptr;
    }
}

const CudaArch* cuda_arch_from_available_hardware() {
    auto options = query_cuda_compute_capabilities();
    if (!options.empty()) {
        return cuda_arch_parse("sm_" + std::to_string(options.front().compute_cap));
    } else {
        return nullptr;
    }
}

const CudaArch* cuda_arch_parse(const std::string& raw_name) {
    // Strip an optional case-insensitive "cuda:" prefix before matching.
    std::string name = raw_name;
    if (name.size() >= 5) {
        std::string prefix = name.substr(0, 5);
        std::transform(prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (prefix == "cuda:") {
            name = name.substr(5);
        }
    }

    if (name == "sm_120") {
        return &CUDA_ARCH_SM120;
    } else if (name == "sm_110") {
        return &CUDA_ARCH_SM110;
    } else if (name == "sm_107") {
        return &CUDA_ARCH_SM107;
    } else if (name == "sm_103") {
        return &CUDA_ARCH_SM103;
    } else if (name == "sm_90") {
        return &CUDA_ARCH_SM90;
    } else if (name == "sm_89") {
        return &CUDA_ARCH_SM89;
    } else if (name == "sm_87") {
        return &CUDA_ARCH_SM87;
    } else if (name == "sm_86") {
        return &CUDA_ARCH_SM86;
    } else if (name == "sm_80") {
        return &CUDA_ARCH_SM80;
    } else if (name == "sm_75") {
        return &CUDA_ARCH_SM75;
    } else if (name == "sm_70") {
        return &CUDA_ARCH_SM70;
    } else {
        return nullptr;
    }
}

CudaArch CUDA_ARCH_SM70{70, false, false, false};
CudaArch CUDA_ARCH_SM75{75, true, false, false};
CudaArch CUDA_ARCH_SM80{80, true, true, true};
CudaArch CUDA_ARCH_SM86{86, true, true, false};
CudaArch CUDA_ARCH_SM87{87, true, true, false};
CudaArch CUDA_ARCH_SM89{89, true, true, false};
CudaArch CUDA_ARCH_SM90{90, true, true, true};
CudaArch CUDA_ARCH_SM100{100, true, true, true};
CudaArch CUDA_ARCH_SM103{103, true, true, false};
CudaArch CUDA_ARCH_SM107{107, true, true, true};
CudaArch CUDA_ARCH_SM110{111, true, true, false};
CudaArch CUDA_ARCH_SM120{120, true, false, false};

} // namespace sdfg::gpu::cuda
