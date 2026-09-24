#include "sdfg/targets/gpu/gpu_arch.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_arch.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_arch.h"

namespace sdfg::gpu {

int GpuMmaSupport::get_integer_block_count(const symbolic::Expression& size, uint16_t block_size) {
    auto blocks = symbolic::simplify(SymEngine::div(size, symbolic::integer(block_size)));
    if (SymEngine::is_a<SymEngine::Integer>(*blocks)) {
        auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(blocks);
        if (i->as_int() > 0) {
            return static_cast<int>(i->as_int());
        }
    }
    return 0;
}

const GpuArch* GpuArch::get_from_schedule_type(const structured_control_flow::ScheduleType& schedule) {
    if (schedule.value() == ::sdfg::cuda::ScheduleType_CUDA_Offload::value()) {
        return cuda::cuda_arch_from_schedule_type(schedule);
    } else if (schedule.value() == ::sdfg::rocm::ScheduleType_ROCM_Offload::value()) {
        return rocm::rocm_arch_from_schedule_type(schedule);
    }
    return nullptr;
}

namespace util {

/// Runs a command and captures its standard output. Returns std::nullopt if the
/// command could not be launched.
std::optional<std::string> run_and_capture(const std::string& command) {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), &pclose);
    if (!pipe) {
        return std::nullopt;
    }

    std::string output;
    std::array<char, 256> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        output += buffer.data();
    }
    return output;
}

/// Trims surrounding whitespace from a string.
std::string trim(const std::string& raw) {
    const auto begin = raw.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = raw.find_last_not_of(" \t\r\n");
    return raw.substr(begin, end - begin + 1);
}

} // namespace util

} // namespace sdfg::gpu
