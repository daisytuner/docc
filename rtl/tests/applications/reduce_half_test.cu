#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuFree hipFree
#else
#include <cuda_runtime.h>
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuFree cudaFree
#endif

#include <cmath>
#include <cstdio>
#include <daisy_rtl/daisy_rtl.h>
#include <iterator>

__global__ void combine_half(_Float16* values, int operation, _Float16 first, _Float16 second) {
    unsigned int slot = (blockIdx.x * blockDim.x + threadIdx.x) % 2;
    _Float16 update = slot == 0 ? first : second;
    switch (operation) {
        case 0:
            __daisy_reduce_combine_add__Float16(values + slot, update);
            break;
        case 1:
            __daisy_reduce_combine_mul__Float16(values + slot, update);
            break;
        case 2:
            __daisy_reduce_combine_min__Float16(values + slot, update);
            break;
        case 3:
            __daisy_reduce_combine_max__Float16(values + slot, update);
            break;
    }
}

_Float16 reference_combine(_Float16 current, _Float16 update, int operation) {
    float left = static_cast<float>(current);
    float right = static_cast<float>(update);
    switch (operation) {
        case 0:
            return static_cast<_Float16>(left + right);
        case 1:
            return static_cast<_Float16>(left * right);
        case 2:
            return left < right ? current : update;
        case 3:
            return left > right ? current : update;
        default:
            return current;
    }
}

int main() {
    int devices = 0;
    if (gpuGetDeviceCount(&devices) != 0 || devices == 0) {
        return 77;
    }
    _Float16* device = nullptr;
    if (gpuMalloc(reinterpret_cast<void**>(&device), 8 * sizeof(_Float16)) != 0) {
        return 1;
    }
    const _Float16 nan = __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x7e00));
    const _Float16 subnormal = __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x0001));
    struct TestCase {
        const char* name;
        int operation;
        _Float16 initial[2];
        _Float16 update[2];
    };
    const TestCase cases[] = {
        {"add fractions", 0, {0, 0}, {0.5, -0.25}},
        {"add rounding ties", 0, {1, 1.0009765625}, {0.00048828125, 0.00048828125}},
        {"add subnormals", 0, {0, 0}, {subnormal, static_cast<_Float16>(-subnormal)}},
        {"add overflow", 0, {65504, -65504}, {32, -32}},
        {"multiply fractions", 1, {2, -3}, {1.0009765625, 0.9990234375}},
        {"multiply underflow", 1, {1, -1}, {0.5, 0.5}},
        {"minimum improves", 2, {10, 11}, {-2, -3.5}},
        {"minimum retains", 2, {-10, -11}, {5.5, 7.25}},
        {"maximum improves", 3, {-10, -11}, {5.5, 7.25}},
        {"maximum retains", 3, {10, 11}, {-2, -3.5}},
        {"minimum signed zero", 2, {0.0, -0.0}, {-0.0, 0.0}},
        {"maximum signed zero", 3, {0.0, -0.0}, {-0.0, 0.0}},
        {"add NaN", 0, {nan, 1}, {0, nan}},
        {"multiply NaN", 1, {nan, 1}, {1, nan}},
    };
    constexpr int blocks = 4;
    constexpr int threads = 64;
    int result = 0;
    for (const auto& test : cases) {
        _Float16 expected[] = {test.initial[0], test.initial[1]};
        for (int update = 0; update < blocks * threads / 2; ++update) {
            for (int slot = 0; slot < 2; ++slot) {
                expected[slot] = reference_combine(expected[slot], test.update[slot], test.operation);
            }
        }
        _Float16 values[] = {
            3.25, test.initial[0], test.initial[1], -7.5, test.initial[0], test.initial[1], 2.5, -6.25
        };
        if (gpuMemcpy(device, values, sizeof(values), gpuMemcpyHostToDevice) != 0) {
            result = 1;
            break;
        }
        for (int offset : {1, 4}) {
            combine_half<<<blocks, threads>>>(device + offset, test.operation, test.update[0], test.update[1]);
            auto launch_error = gpuGetLastError();
            if (launch_error != 0) {
                std::fprintf(stderr, "%s launch failed: %s\n", test.name, gpuGetErrorString(launch_error));
                result = 1;
                break;
            }
        }
        if (result != 0) {
            break;
        }
        if (gpuDeviceSynchronize() != 0 || gpuMemcpy(values, device, sizeof(values), gpuMemcpyDeviceToHost) != 0) {
            result = 1;
            break;
        }
        if (values[0] != 3.25 || values[3] != -7.5 || values[6] != 2.5 || values[7] != -6.25) {
            std::fprintf(stderr, "Neighboring half overwritten for %s\n", test.name);
            result = 1;
        }
        for (int offset : {1, 4}) {
            for (int slot = 0; slot < 2; ++slot) {
                float actual = static_cast<float>(values[offset + slot]);
                float wanted = static_cast<float>(expected[slot]);
                auto actual_bits = __builtin_bit_cast(unsigned short, values[offset + slot]);
                auto expected_bits = __builtin_bit_cast(unsigned short, expected[slot]);
                if (!(actual_bits == expected_bits || (std::isnan(actual) && std::isnan(wanted)))) {
                    std::fprintf(
                        stderr,
                        "%s, offset %d, slot %d: %g (0x%04x) != %g (0x%04x)\n",
                        test.name,
                        offset,
                        slot,
                        actual,
                        static_cast<unsigned int>(actual_bits),
                        wanted,
                        static_cast<unsigned int>(expected_bits)
                    );
                    result = 1;
                }
            }
        }
    }
    if (gpuFree(device) != 0) {
        return 1;
    }
    if (result == 0) {
        std::printf("Passed %zu FP16 numerical cases\n", std::size(cases));
    }
    return result;
}
