/**
 * @file type.h
 * @brief Core type system definitions for sdfglib
 *
 * This file defines the fundamental types used throughout sdfglib, including
 * primitive types, storage types, and the base type interface.
 */

#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <type_traits>

#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

namespace types {

/**
 * @enum PrimitiveType
 * @brief Enumeration of all primitive data types supported by sdfglib
 *
 * This enum defines all the basic scalar types that can be used in the type system,
 * including integer types (signed and unsigned), floating-point types, and void.
 */
enum PrimitiveType {
    Void, ///< Void type (0 bits)
    Bool, ///< Boolean type (1 bit)
    Int8, ///< 8-bit signed integer
    Int16, ///< 16-bit signed integer
    Int32, ///< 32-bit signed integer
    Int64, ///< 64-bit signed integer
    Int128, ///< 128-bit signed integer
    UInt8, ///< 8-bit unsigned integer
    UInt16, ///< 16-bit unsigned integer
    UInt32, ///< 32-bit unsigned integer
    UInt64, ///< 64-bit unsigned integer
    UInt128, ///< 128-bit unsigned integer
    Half, ///< 16-bit floating-point (IEEE 754 half precision)
    BFloat, ///< 16-bit brain floating-point
    Float, ///< 32-bit floating-point (IEEE 754 single precision)
    Double, ///< 64-bit floating-point (IEEE 754 double precision)
    X86_FP80, ///< 80-bit extended precision floating-point (x86)
    FP128, ///< 128-bit floating-point (IEEE 754 quadruple precision)
    PPC_FP128 ///< 128-bit floating-point (PowerPC double-double)
};

/**
 * @class StorageType
 * @brief Represents the storage location and management characteristics of data
 *
 * StorageType defines where data is stored (e.g., CPU stack, CPU heap, GPU memory)
 * and how its allocation and deallocation are managed.
 */
class StorageType {
public:
    /**
     * @enum AllocationType
     * @brief Defines how memory allocation/deallocation is managed
     */
    enum AllocationType {
        Unmanaged, ///< Memory is not automatically managed
        Managed, ///< Memory is automatically managed
    };

private:
    std::string value_; ///< Storage location identifier
    symbolic::Expression allocation_size_; ///< Size to allocate
    AllocationType allocation_; ///< Allocation management type
    AllocationType deallocation_; ///< Deallocation management type
    symbolic::Expression arg1_; ///< Additional argument (e.g., page_size for specialized hardware)

public:
    /**
     * @brief Constructs a StorageType with default unmanaged allocation
     * @param value The storage location identifier
     */
    StorageType(const std::string& value)
        : value_(value), allocation_size_(SymEngine::null), allocation_(Unmanaged), deallocation_(Unmanaged) {}

    /**
     * @brief Constructs a StorageType with explicit allocation management
     * @param value The storage location identifier
     * @param allocation_size The size to allocate
     * @param allocation The allocation management type
     * @param deallocation The deallocation management type
     */
    StorageType(
        const std::string& value,
        const symbolic::Expression& allocation_size,
        AllocationType allocation,
        AllocationType deallocation
    )
        : value_(value), allocation_size_(allocation_size), allocation_(allocation), deallocation_(deallocation) {}

    /// @brief Gets the storage location identifier
    std::string value() const { return value_; }

    /// @brief Sets the storage location identifier
    void value(const std::string& value) { value_ = value; }

    /// @brief Gets the allocation size
    symbolic::Expression allocation_size() const { return allocation_size_; }

    /// @brief Sets the allocation size
    void allocation_size(const symbolic::Expression& allocation_size) { allocation_size_ = allocation_size; }

    /// @brief Gets the additional argument
    symbolic::Expression arg1() const { return arg1_; }

    /// @brief Sets the additional argument
    void arg1(const symbolic::Expression& arg) { arg1_ = arg; }

    /// @brief Gets the allocation management type
    AllocationType allocation() const { return allocation_; }

    /// @brief Sets the allocation management type
    void allocation(AllocationType allocation) { allocation_ = allocation; }

    /// @brief Gets the deallocation management type
    AllocationType deallocation() const { return deallocation_; }

    /// @brief Sets the deallocation management type
    void deallocation(AllocationType deallocation) { deallocation_ = deallocation; }

    /**
     * @brief Compares two StorageType objects for equality
     * @param other The StorageType to compare with
     * @return true if the storage types are equal, false otherwise
     */
    bool operator==(const StorageType& other) const {
        if (value_ != other.value_) {
            return false;
        }
        if (allocation_ != other.allocation_) {
            return false;
        }
        if (deallocation_ != other.deallocation_) {
            return false;
        }
        if (!symbolic::null_safe_eq(allocation_size_, other.allocation_size_)) {
            return false;
        }
        if (!symbolic::null_safe_eq(arg1_, other.arg1_)) {
            return false;
        }
        return true;
    }

    /// @brief Checks if this storage is CPU stack storage
    bool is_cpu_stack() const { return value_ == "CPU_Stack"; }

    /// @brief Checks if this storage is CPU heap storage
    bool is_cpu_heap() const { return value_ == "CPU_Heap"; }

    /// @brief Checks if this storage resides on a device (i.e. not CPU stack/heap)
    bool is_device() const { return !is_cpu_stack() && !is_cpu_heap(); }

    /// @brief Checks if this storage is NVIDIA generic memory
    bool is_nv_generic() const { return value_ == "NV_Generic"; }

    /// @brief Checks if this storage is NVIDIA global memory
    bool is_nv_global() const { return value_ == "NV_Global"; }

    /// @brief Checks if this storage is NVIDIA shared memory
    bool is_nv_shared() const { return value_ == "NV_Shared"; }

    /// @brief Checks if this storage is NVIDIA constant memory
    bool is_nv_constant() const { return value_ == "NV_Constant"; }

    /// @brief Checks if this storage is an NVIDIA symbol
    bool is_nv_symbol() const { return value_ == "NV_Symbol"; }

    /**
     * @brief Creates a CPU stack storage type
     * @return StorageType configured for CPU stack with unmanaged allocation
     */
    static StorageType CPU_Stack() {
        return StorageType("CPU_Stack", SymEngine::null, StorageType::Unmanaged, StorageType::Unmanaged);
    }

    /**
     * @brief Creates a CPU heap storage type with default unmanaged allocation
     * @return StorageType configured for CPU heap with unmanaged allocation
     */
    static StorageType CPU_Heap() {
        return StorageType("CPU_Heap", SymEngine::null, StorageType::Unmanaged, StorageType::Unmanaged);
    }

    /**
     * @brief Creates a CPU heap storage type with explicit allocation management
     * @param allocation_size The size to allocate
     * @param allocation The allocation management type
     * @param deallocation The deallocation management type
     * @return StorageType configured for CPU heap with specified allocation management
     */
    static StorageType CPU_Heap(
        symbolic::Expression allocation_size,
        StorageType::AllocationType allocation,
        StorageType::AllocationType deallocation
    ) {
        return StorageType("CPU_Heap", allocation_size, allocation, deallocation);
    }

    /// @brief Creates an NVIDIA generic memory storage type
    static StorageType NV_Generic() { return StorageType("NV_Generic"); }

    /// @brief Creates an NVIDIA global memory storage type
    static StorageType NV_Global() { return StorageType("NV_Global"); }

    /// @brief Creates an NVIDIA shared memory storage type
    static StorageType NV_Shared() { return StorageType("NV_Shared"); }

    /// @brief Creates an NVIDIA constant memory storage type
    static StorageType NV_Constant() { return StorageType("NV_Constant"); }

    /// @brief Creates an NVIDIA symbol storage type
    static StorageType NV_Symbol() { return StorageType("NV_Symbol"); }
};

/**
 * @brief Converts a PrimitiveType to its string representation
 * @param e The PrimitiveType to convert
 * @return The string name of the primitive type
 * @throws std::invalid_argument if the primitive type is invalid
 */
constexpr const char* primitive_type_to_string(PrimitiveType e) {
    switch (e) {
        case PrimitiveType::Void:
            return "Void";
        case PrimitiveType::Bool:
            return "Bool";
        case PrimitiveType::Int8:
            return "Int8";
        case PrimitiveType::Int16:
            return "Int16";
        case PrimitiveType::Int32:
            return "Int32";
        case PrimitiveType::Int64:
            return "Int64";
        case PrimitiveType::Int128:
            return "Int128";
        case PrimitiveType::UInt8:
            return "UInt8";
        case PrimitiveType::UInt16:
            return "UInt16";
        case PrimitiveType::UInt32:
            return "UInt32";
        case PrimitiveType::UInt64:
            return "UInt64";
        case PrimitiveType::UInt128:
            return "UInt128";
        case PrimitiveType::Half:
            return "Half";
        case PrimitiveType::BFloat:
            return "BFloat";
        case PrimitiveType::Float:
            return "Float";
        case PrimitiveType::Double:
            return "Double";
        case PrimitiveType::X86_FP80:
            return "X86_FP80";
        case PrimitiveType::FP128:
            return "FP128";
        case PrimitiveType::PPC_FP128:
            return "PPC_FP128";
    }
    throw std::invalid_argument("Invalid primitive type");
};

/**
 * @brief Converts a string to its corresponding PrimitiveType
 * @param e The string representation of the primitive type
 * @return The corresponding PrimitiveType enum value
 * @throws std::invalid_argument if the string does not match any primitive type
 */
constexpr PrimitiveType primitive_type_from_string(std::string_view e) {
    if (e == "Void") {
        return PrimitiveType::Void;
    } else if (e == "Bool") {
        return PrimitiveType::Bool;
    } else if (e == "Int8") {
        return PrimitiveType::Int8;
    } else if (e == "Int16") {
        return PrimitiveType::Int16;
    } else if (e == "Int32") {
        return PrimitiveType::Int32;
    } else if (e == "Int64") {
        return PrimitiveType::Int64;
    } else if (e == "Int128") {
        return PrimitiveType::Int128;
    } else if (e == "UInt8") {
        return PrimitiveType::UInt8;
    } else if (e == "UInt16") {
        return PrimitiveType::UInt16;
    } else if (e == "UInt32") {
        return PrimitiveType::UInt32;
    } else if (e == "UInt64") {
        return PrimitiveType::UInt64;
    } else if (e == "UInt128") {
        return PrimitiveType::UInt128;
    } else if (e == "Half") {
        return PrimitiveType::Half;
    } else if (e == "BFloat") {
        return PrimitiveType::BFloat;
    } else if (e == "Float") {
        return PrimitiveType::Float;
    } else if (e == "Double") {
        return PrimitiveType::Double;
    } else if (e == "X86_FP80") {
        return PrimitiveType::X86_FP80;
    } else if (e == "FP128") {
        return PrimitiveType::FP128;
    } else if (e == "PPC_FP128") {
        return PrimitiveType::PPC_FP128;
    }
    throw std::invalid_argument("Invalid primitive type");
};

/**
 * @brief Returns the bit width of a PrimitiveType
 * @param e The PrimitiveType to query
 * @return The bit width of the type (0 for Void, 1 for Bool, etc.)
 * @throws std::invalid_argument if the primitive type is invalid
 */
constexpr size_t bit_width(PrimitiveType e) {
    switch (e) {
        case PrimitiveType::Void:
            return 0;
        case PrimitiveType::Bool:
            return 1;
        case PrimitiveType::Int8:
            return 8;
        case PrimitiveType::Int16:
            return 16;
        case PrimitiveType::Int32:
            return 32;
        case PrimitiveType::Int64:
            return 64;
        case PrimitiveType::Int128:
            return 128;
        case PrimitiveType::UInt8:
            return 8;
        case PrimitiveType::UInt16:
            return 16;
        case PrimitiveType::UInt32:
            return 32;
        case PrimitiveType::UInt64:
            return 64;
        case PrimitiveType::UInt128:
            return 128;
        case PrimitiveType::Half:
            return 16;
        case PrimitiveType::BFloat:
            return 16;
        case PrimitiveType::Float:
            return 32;
        case PrimitiveType::Double:
            return 64;
        case PrimitiveType::X86_FP80:
            return 80;
        case PrimitiveType::FP128:
            return 128;
        case PrimitiveType::PPC_FP128:
            return 128;
    }
    throw std::invalid_argument("Invalid primitive type");
};

/**
 * @brief Checks if a PrimitiveType is a floating-point type
 * @param e The PrimitiveType to check
 * @return true if the type is a floating-point type, false otherwise
 */
constexpr bool is_floating_point(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Half:
        case PrimitiveType::BFloat:
        case PrimitiveType::Float:
        case PrimitiveType::Double:
        case PrimitiveType::X86_FP80:
        case PrimitiveType::FP128:
        case PrimitiveType::PPC_FP128:
            return true;
        default:
            return false;
    }
};

/**
 * @brief Checks if a PrimitiveType is an integer type
 * @param e The PrimitiveType to check
 * @return true if the type is an integer type (including Bool), false otherwise
 */
constexpr bool is_integer(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Bool:
        case PrimitiveType::Int8:
        case PrimitiveType::Int16:
        case PrimitiveType::Int32:
        case PrimitiveType::Int64:
        case PrimitiveType::Int128:
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
        case PrimitiveType::UInt128:
            return true;
        default:
            return false;
    }
};

/**
 * @brief Checks if a PrimitiveType is a signed integer type
 * @param e The PrimitiveType to check
 * @return true if the type is a signed integer type, false otherwise
 */
constexpr bool is_signed(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Int8:
        case PrimitiveType::Int16:
        case PrimitiveType::Int32:
        case PrimitiveType::Int64:
        case PrimitiveType::Int128:
            return true;
        default:
            return false;
    }
};

/**
 * @brief Checks if a PrimitiveType is an unsigned integer type
 * @param e The PrimitiveType to check
 * @return true if the type is an unsigned integer type, false otherwise
 */
constexpr bool is_unsigned(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
        case PrimitiveType::UInt128:
            return true;
        default:
            return false;
    }
};

/**
 * @brief Converts an unsigned integer type to its signed equivalent
 * @param e The PrimitiveType to convert
 * @return The signed equivalent type, or the same type if not unsigned integer
 */
constexpr PrimitiveType as_signed(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::UInt8:
            return PrimitiveType::Int8;
        case PrimitiveType::UInt16:
            return PrimitiveType::Int16;
        case PrimitiveType::UInt32:
            return PrimitiveType::Int32;
        case PrimitiveType::UInt64:
            return PrimitiveType::Int64;
        case PrimitiveType::UInt128:
            return PrimitiveType::Int128;
        default:
            return e;
    }
};

/**
 * @brief Converts a signed integer type to its unsigned equivalent
 * @param e The PrimitiveType to convert
 * @return The unsigned equivalent type, or the same type if not signed integer
 */
constexpr PrimitiveType as_unsigned(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Int8:
            return PrimitiveType::UInt8;
        case PrimitiveType::Int16:
            return PrimitiveType::UInt16;
        case PrimitiveType::Int32:
            return PrimitiveType::UInt32;
        case PrimitiveType::Int64:
            return PrimitiveType::UInt64;
        case PrimitiveType::Int128:
            return PrimitiveType::UInt128;
        default:
            return e;
    }
};

/**
 * @enum TypeID
 * @brief Enumeration of high-level type categories
 *
 * This enum is used for runtime type identification of IType-derived classes.
 */
enum class TypeID {
    Scalar, ///< Scalar primitive type
    Array, ///< Array type (c-style arrays)
    Structure, ///< Structure/record type
    Pointer, ///< Pointer type
    Reference, ///< Reference type
    Function, ///< Function type
    Tensor, ///< Tensor type (multi-dimensional arrays)
};

/**
 * @class IType
 * @brief Abstract base interface for all types in the sdfglib type system
 *
 * IType provides the common interface and properties for all type representations
 * in sdfglib. All concrete type classes (Scalar, Array, Pointer, etc.) inherit
 * from this interface.
 */
class IType {
protected:
    StorageType storage_type_; ///< Where and how the data is stored
    size_t alignment_; ///< Memory alignment requirement in bytes
    std::string initializer_; ///< Optional initializer expression

public:
    /**
     * @brief Constructs an IType with optional storage, alignment, and initializer
     * @param storage_type The storage location and management (default: CPU_Stack)
     * @param alignment Memory alignment in bytes (default: 0 for natural alignment)
     * @param initializer Optional initializer expression (default: empty)
     */
    IType(StorageType storage_type = StorageType::CPU_Stack(), size_t alignment = 0, const std::string& initializer = "")
        : storage_type_(storage_type), alignment_(alignment), initializer_(initializer) {};

    /// Virtual destructor for proper cleanup of derived classes
    virtual ~IType() = default;

    /**
     * @brief Returns the TypeID for runtime type identification
     * @return The TypeID enum value for this type
     */
    virtual TypeID type_id() const = 0;

    /// @brief Gets the storage type (const version)
    StorageType storage_type() const { return storage_type_; }

    /**
     * @brief Checks if this is a pointer or a structure behaves like a pointer
     *
     * Some structures may have pointer-like semantics in certain contexts.
     */
    virtual bool is_pointer_like() const = 0;

    /// @brief Gets the storage type (mutable reference)
    StorageType& storage_type() { return storage_type_; };

    /// @brief Sets the storage type
    void storage_type(const StorageType& storage_type) { storage_type_ = storage_type; };

    /// @brief Gets the alignment requirement in bytes
    size_t alignment() const { return alignment_; };

    /// @brief Sets the alignment requirement in bytes
    void alignment(size_t alignment) { alignment_ = alignment; };

    /// @brief Gets the initializer expression
    std::string initializer() const { return initializer_; };

    /// @brief Sets the initializer expression
    void initializer(const std::string& initializer) { initializer_ = initializer; };

    /**
     * @brief Returns the primitive type for this type
     *
     * For scalar types, this returns the actual primitive type.
     * For composite types (arrays, pointers, etc.), this returns the primitive
     * type of the innermost element.
     *
     * @return The primitive type
     */
    virtual PrimitiveType primitive_type() const = 0;

    /**
     * @brief Checks if this type represents a symbolic/integer type
     *
     * Symbolic types are integer types that can be used in symbolic expressions
     * and loop bounds.
     *
     * @return true if this is a symbolic integer type, false otherwise
     */
    virtual bool is_symbol() const = 0;

    /**
     * @brief Compares two types for equality
     * @param other The type to compare with
     * @return true if the types are equal, false otherwise
     */
    virtual bool operator==(const IType& other) const = 0;

    /**
     * @brief Creates a deep copy of this type
     * @return A unique pointer to a new type object with the same properties
     */
    virtual std::unique_ptr<IType> clone() const = 0;

    /**
     * @brief Returns a string representation of this type
     * @return A human-readable string describing this type
     */
    virtual std::string print() const = 0;

    /**
     * @brief Stream output operator for types
     * @param os The output stream
     * @param type The type to output
     * @return The output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const IType& type) {
        os << type.print();
        return os;
    };
};

} // namespace types
} // namespace sdfg
