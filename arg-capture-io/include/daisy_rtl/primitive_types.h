#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace arg_capture {

enum class PrimitiveType {
    Void = 0,
    Bool = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    Int128 = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    UInt128 = 11,
    Half = 12,
    BFloat = 13,
    Float = 14,
    Double = 15,
    X86_FP80 = 16,
    FP128 = 17,
    PPC_FP128 = 18,
    PRIMITIVE_TYPE_COUNT = 19
};

static const char* primitive_type_names[] = {
    "Void",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "Half",
    "BFloat",
    "Float",
    "Double",
    "X86_FP80",
    "FP128",
    "PPC_FP128"
};

static const char* primitive_type_cppspellings[] = {
    "void", // Void
    "bool", // Bool
    "int8_t", // Int8
    "int16_t", // Int16
    "int32_t", // Int32
    "int64_t", // Int64
    "__int128", // Int128
    "uint8_t", // UInt8
    "uint16_t", // UInt16
    "uint32_t", // UInt32
    "uint64_t", // UInt64
    "unsigned __int128", // UInt128
    "_Float16", // Half
    "__bf16", // BFloat (clang builtin, 16-bit IEEE bfloat16; storage-only, promotes to float)
    "float", // Float
    "double", // Double
    "long double", // X86_FP80  (x86 80-bit extended precision)
    "__float128", // FP128     (software quad precision via libquadmath)
    "__ibm128" // PPC_FP128 (IBM double-double)
};

static const double primitive_type_machine_epsilons[] = {
    0.0, // void
    0.0, // bool
    0.0, // int8_t
    0.0, // int16_t
    0.0, // int32_t
    0.0, // int64_t
    0.0, // __int128
    0.0, // uint8_t
    0.0, // uint16_t
    0.0, // uint32_t
    0.0, // uint64_t
    0.0, // unsigned __int128
    9.765625e-04, // Half (_Float16)  ≈ 2^-10
    7.8125e-03, // BFloat16 (__bf16) ≈ 2^-7
    1.1920929e-07, // Float (float, 32-bit)  ≈ 2^-23
    2.220446049250313e-16, // Double (double, 64-bit) ≈ 2^-52
    1.0842021724855044e-19, // long double (x86_80-bit) ≈ 2^-63
    1.9259299443872354e-34, // __float128 (quad-precision) ≈ 2^-112
    1.0842021724855044e-19 // __ibm128 (double-double) ≈ 2^-63
};

constexpr const char* to_string(PrimitiveType e) {
    if (e < PrimitiveType::Void || e >= PrimitiveType::PRIMITIVE_TYPE_COUNT) {
        return "[unknown]";
    } else {
        return primitive_type_names[static_cast<int32_t>(e)];
    }
}

// ----------------------------
// Template to get C++ type from PrimitiveType
// ----------------------------
template<PrimitiveType T>
struct PrimitiveCppType;

// Floating-point types
template<>
struct PrimitiveCppType<PrimitiveType::Float> {
    using type = float;
};
template<>
struct PrimitiveCppType<PrimitiveType::Double> {
    using type = double;
};
template<>
struct PrimitiveCppType<PrimitiveType::X86_FP80> {
    using type = long double;
};

// Integral types
template<>
struct PrimitiveCppType<PrimitiveType::Int8> {
    using type = int8_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::Int16> {
    using type = int16_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::Int32> {
    using type = int32_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::Int64> {
    using type = int64_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::UInt8> {
    using type = uint8_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::UInt16> {
    using type = uint16_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::UInt32> {
    using type = uint32_t;
};
template<>
struct PrimitiveCppType<PrimitiveType::UInt64> {
    using type = uint64_t;
};

// ----------------------------
// Concept for types with numeric_limits
// ----------------------------
template<typename T>
concept HasNumericLimits = requires { std::numeric_limits<T>::is_specialized; } &&
                           std::numeric_limits<T>::is_specialized;

// Machine epsilon for any type with numeric_limits
template<HasNumericLimits T>
constexpr T machine_epsilon() {
    return std::numeric_limits<T>::epsilon();
}

// Fallback for unsupported types triggers compile-time error
template<typename T>
constexpr T machine_epsilon() {
    throw std::runtime_error("machine_epsilon not defined for this type");
    return T{};
}

inline long double primitive_type_epsilon(arg_capture::PrimitiveType pt) {
    switch (pt) {
        // Single source of truth: the machine-epsilon table (there is no portable
        // numeric_limits specialization for the 16-bit floats).
        case arg_capture::PrimitiveType::Half:
        case arg_capture::PrimitiveType::BFloat:
        case arg_capture::PrimitiveType::Float:
        case arg_capture::PrimitiveType::Double:
        case arg_capture::PrimitiveType::X86_FP80:
        case arg_capture::PrimitiveType::FP128:
        case arg_capture::PrimitiveType::PPC_FP128:
            return static_cast<long double>(primitive_type_machine_epsilons[static_cast<int32_t>(pt)]);
        default:
            throw std::runtime_error("No machine epsilon defined for this type");
    }
}

} // namespace arg_capture
