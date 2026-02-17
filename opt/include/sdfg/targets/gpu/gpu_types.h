#pragma once

namespace sdfg {
namespace gpu {

/**
 * @brief Shared GPU dimension enum for CUDA/HIP targets
 * Represents the X, Y, Z dimensions used for GPU kernel launches
 */
enum GPUDimension { X = 0, Y = 1, Z = 2 };

} // namespace gpu
} // namespace sdfg
