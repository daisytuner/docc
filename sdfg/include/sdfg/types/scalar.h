/**
 * @file scalar.h
 * @brief Scalar (primitive) type representation
 *
 * This file defines the Scalar class, which represents primitive scalar types
 * such as integers, floating-point numbers, and booleans.
 */

#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

/**
 * @class Scalar
 * @brief Represents a scalar primitive type
 *
 * The Scalar class wraps a PrimitiveType and provides operations for working
 * with scalar types, including type conversions between signed and unsigned variants.
 *
 * Scalar types are the most basic types in the type system and include integers,
 * floating-point numbers, and booleans.
 */
class Scalar : public IType {
private:
    PrimitiveType primitive_type_; ///< The underlying primitive type

public:
    /**
     * @brief Constructs a Scalar with the specified primitive type
     * @param primitive_type The primitive type (e.g., Int32, Float, Bool)
     */
    Scalar(PrimitiveType primitive_type);

    /**
     * @brief Constructs a Scalar with explicit storage, alignment, initializer, and primitive type
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     * @param primitive_type The primitive type
     */
    Scalar(StorageType storage_type, size_t alignment, const std::string& initializer, PrimitiveType primitive_type);

    /// @brief Returns TypeID::Scalar
    virtual TypeID type_id() const override;

    /// @brief Returns the primitive type of this scalar
    virtual PrimitiveType primitive_type() const override;

    /**
     * @brief Checks if this scalar type can be used as a symbol
     *
     * Integer types (signed and unsigned) are considered symbolic and can be used
     * in symbolic expressions. Floating-point types are not symbolic.
     *
     * @return true if this is an integer type, false otherwise
     */
    virtual bool is_symbol() const override;

    bool is_pointer_like() const override { return false; }

    /**
     * @brief Creates a copy of this scalar with signed integer type
     *
     * If this scalar is an unsigned integer type, returns a new Scalar with the
     * corresponding signed type. Otherwise, returns a copy with the same type.
     *
     * @return A new Scalar with signed type
     */
    Scalar as_signed() const;

    /**
     * @brief Creates a copy of this scalar with unsigned integer type
     *
     * If this scalar is a signed integer type, returns a new Scalar with the
     * corresponding unsigned type. Otherwise, returns a copy with the same type.
     *
     * @return A new Scalar with unsigned type
     */
    Scalar as_unsigned() const;

    /// @brief Compares this scalar with another type for equality
    virtual bool operator==(const IType& other) const override;

    /// @brief Creates a deep copy of this scalar
    virtual std::unique_ptr<IType> clone() const override;

    /// @brief Returns a string representation of this scalar type
    virtual std::string print() const override;

    /**
     * @brief Replace symbolic expressions on this type
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     *
     * Nothing to do.
     */
    virtual void replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression)
        override {}
    virtual void replace_symbols(const symbolic::ExpressionMapping& replacements) override {}
};
} // namespace types
} // namespace sdfg
