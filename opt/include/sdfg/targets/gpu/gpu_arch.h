#pragma once

#include <stdint.h>
#include <string>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {
class GpuArch;

struct GpuMmaTiling {
    int mma_block_m = 0;
    int mma_block_n = 0;
    int mma_block_k = 0;
    int wave_tile_blocks_m = 0;
    int wave_tile_blocks_n = 0;
    int threads_per_mma_block_m = 0;
    int macro_blocks_m = 0;
    int macro_blocks_n = 0;
};

struct GpuMmaSupport {
    const uint16_t mma_block_m;
    const uint16_t mma_block_n;
    const uint16_t mma_block_k;
    const uint16_t threads_per_mma_block;

    GpuMmaSupport(uint16_t block_m, uint16_t block_n, uint16_t block_k, uint16_t threads_per_mma_block)
        : mma_block_m(block_m), mma_block_n(block_n), mma_block_k(block_k),
          threads_per_mma_block(threads_per_mma_block) {}
    virtual bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const = 0;

    virtual std::optional<data_flow::ImplementationType>
    get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling) const = 0;

    static int get_integer_block_count(const symbolic::Expression& size, uint16_t block_size);

    virtual bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const = 0;

    virtual GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const = 0;
};

class GpuArch {
public:
    static constexpr const char* ARCH_PROPERTY = "ARCH";

    GpuArch() {}
    virtual ~GpuArch() = default;

    virtual std::string unique_id() const = 0;
    virtual std::string name() const = 0;
    virtual int per_cu_threads() const = 0;

    virtual const GpuMmaSupport* mma_support() const = 0;

    virtual structured_control_flow::ScheduleType create_schedule_type() const = 0;

    static const GpuArch* get_from_schedule_type(const structured_control_flow::ScheduleType& schedule);
};

namespace util {

/// Runs a command and captures its standard output. Returns std::nullopt if the
/// command could not be launched.
std::optional<std::string> run_and_capture(const std::string& command);

/// Trims surrounding whitespace from a string.
std::string trim(const std::string& raw);

} // namespace util

} // namespace sdfg::gpu
