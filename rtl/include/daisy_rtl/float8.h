#pragma once

// FP8 values for generated C++, which has no FP8 type: one byte of storage, computed with as float. Reading one
// converts it to float exactly; storing a float rounds it to the nearest, ties to even, as ml_dtypes does. Out of
// range, E4M3 (finite only, "e4m3fn") becomes NaN and E5M2 infinity.

#ifdef __cplusplus

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace daisy_rtl {

/// An 8-bit float of @p E exponent and @p M mantissa bits, of IEEE layout; @p Finite types have no infinities and
/// spend only the all-ones pattern on NaN.
template<int E, int M, bool Finite>
struct Float8 {
    static constexpr int bias = (1 << (E - 1)) - 1;
    static constexpr int max_biased = (1 << E) - 1;

    std::uint8_t bits;

    Float8() = default;

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    Float8(T value) : bits(encode(static_cast<float>(value))) {
    }

    operator float() const {
        return decode(bits);
    }

    static float decode(std::uint8_t bits) {
        float sign = (bits & 0x80) ? -1.0f : 1.0f;
        int biased = (bits >> M) & max_biased;
        int mantissa = bits & ((1 << M) - 1);
        if (Finite ? (biased == max_biased && mantissa == (1 << M) - 1) : (biased == max_biased && mantissa != 0)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (!Finite && biased == max_biased) {
            return sign * std::numeric_limits<float>::infinity();
        }
        if (biased == 0) {
            return sign * std::ldexp(static_cast<float>(mantissa), 1 - bias - M);
        }
        return sign * std::ldexp(static_cast<float>(mantissa + (1 << M)), biased - bias - M);
    }

    static std::uint8_t encode(float value) {
        constexpr std::uint8_t nan = Finite ? 0x7F : static_cast<std::uint8_t>((max_biased << M) | 1);
        constexpr std::uint8_t infinity = static_cast<std::uint8_t>(max_biased << M);
        std::uint8_t sign = std::signbit(value) ? 0x80 : 0;
        if (std::isnan(value)) {
            return nan;
        }
        float magnitude = std::fabs(value);
        if (std::isinf(magnitude)) {
            return Finite ? nan : static_cast<std::uint8_t>(sign | infinity);
        }
        if (magnitude == 0.0f) {
            return sign;
        }
        // The exponent of the leading bit, at least the smallest normal's; the significand then rounds to M bits
        // after it, ties to even (the default rounding mode), subnormals alike.
        int exponent = std::max(std::ilogb(magnitude), 1 - bias);
        float rounded = std::nearbyint(std::ldexp(magnitude, M - exponent));
        if (rounded >= static_cast<float>(2 << M)) {
            rounded /= 2;
            exponent += 1;
        }
        int significand = static_cast<int>(rounded);
        int biased = significand >= (1 << M) ? exponent + bias : 0;
        int mantissa = significand & ((1 << M) - 1);
        bool overflow = Finite ? (biased > max_biased || (biased == max_biased && mantissa == (1 << M) - 1))
                               : biased >= max_biased;
        if (overflow) {
            return Finite ? nan : static_cast<std::uint8_t>(sign | infinity);
        }
        return static_cast<std::uint8_t>(sign | (biased << M) | mantissa);
    }
};

} // namespace daisy_rtl

using __daisy_float8_e4m3 = daisy_rtl::Float8<4, 3, true>;
using __daisy_float8_e5m2 = daisy_rtl::Float8<5, 2, false>;

static_assert(sizeof(__daisy_float8_e4m3) == 1 && sizeof(__daisy_float8_e5m2) == 1);

#endif
