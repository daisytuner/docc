/**
 * @file pointer.h
 * @brief Pointer type representation
 *
 * This file defines the Pointer class, which represents pointer types that
 * reference other types in memory.
 */

#pragma once

#include <optional>

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

/**
 * @class Pointer
 * @brief Represents a pointer type
 *
 * The Pointer class represents a pointer to another type. Pointers can be:
 * - Typed: pointing to a specific type (e.g., `int*`, `float*`)
 * - Opaque: pointing to an unknown or void type (e.g., `void*`)
 *
 * Pointers are fundamental for representing memory addresses, dynamic data structures,
 * and references to data in different memory spaces (CPU heap, GPU memory, etc.).
 *
 * @note The pointee type is optional to support void pointers and opaque pointers.
 */
class Pointer : public IType {
private:
    std::optional<std::unique_ptr<IType>> pointee_type_; ///< Type pointed to (optional for void*)

public:
    /**
     * @brief Constructs an opaque pointer (void* equivalent)
     *
     * Creates a pointer without a specific pointee type, similar to void* in C/C++.
     */
    Pointer();

    /**
     * @brief Constructs a typed pointer to the specified type
     *
     * WARNING: This constructor is less specific than the COPY-constructor,
     * which still EXISTS and behaves differently!
     *
     * @param pointee_type The type of the object pointed to by this pointer
     */
    Pointer(const IType& pointee_type);

    /**
     * @brief Constructs an opaque pointer with explicit storage and alignment
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     */
    Pointer(StorageType storage_type, size_t alignment, const std::string& initializer);

    /**
     * @brief Constructs a typed pointer with explicit storage, alignment, and pointee type
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     * @param pointee_type The type of the object pointed to by this pointer
     */
    Pointer(StorageType storage_type, size_t alignment, const std::string& initializer, const IType& pointee_type);

    /// @brief Creates a deep copy of this pointer type
    virtual std::unique_ptr<IType> clone() const override;

    /// @brief Returns TypeID::Pointer
    virtual TypeID type_id() const override;

    /**
     * @brief Returns the primitive type of the pointee
     *
     * For typed pointers, returns the primitive type of the pointed-to object.
     * For opaque pointers, the behavior depends on the implementation.
     *
     * @return The primitive type of the pointee
     */
    virtual PrimitiveType primitive_type() const override;

    /**
     * @brief Pointers are symbolic types
     * @return Always returns true
     */
    virtual bool is_symbol() const override;

    bool is_pointer_like() const override { return true; }

    /**
     * @brief Checks if this pointer has a known pointee type
     * @return true if the pointee type is specified, false for opaque pointers
     */
    bool has_pointee_type() const;

    /**
     * @brief Gets the type of the object pointed to
     * @return A const reference to the pointee type
     * @pre has_pointee_type() must return true
     */
    const IType& pointee_type() const;

    /// @brief Compares this pointer with another type for equality
    virtual bool operator==(const IType& other) const override;

    /// @brief Returns a string representation of this pointer type
    virtual std::string print() const override;

    /**
     * @brief Replace symbolic expressions on this type
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     *
     * Replaces occurrences of symbolic expressions on the type.
     */
    virtual void replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression)
        override;
    virtual void replace_symbols(const symbolic::ExpressionMapping& replacements) override;
};
} // namespace types
} // namespace sdfg
