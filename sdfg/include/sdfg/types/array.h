/**
 * @file array.h
 * @brief Array type representation for c-style arrays
 *
 * This file defines the Array class, which represents fixed-size arrays
 * in the C/C++ style (not std::array). Arrays have a known element type
 * and a size that can be determined either at compile-time or symbolically.
 */

#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

/**
 * @class Array
 * @brief Represents a c-style array type
 *
 * The Array class represents a fixed-size sequence of elements of the same type.
 * This corresponds to C-style arrays like `int[10]` or `float[N]` where N is
 * a symbolic expression.
 *
 * Arrays can be nested to create multi-dimensional arrays. The number of elements
 * can be specified as either a concrete integer or a symbolic expression that may
 * depend on parameters or variables.
 *
 * @note This represents c-style arrays, not C++ std::array or std::vector.
 * @note Arrays are stored contiguously in memory according to the storage type.
 */
class Array : public IType {
private:
    std::unique_ptr<IType> element_type_; ///< Type of each array element
    symbolic::Expression num_elements_; ///< Number of elements (can be symbolic)

public:
    /**
     * @brief Constructs an Array with the specified element type and size
     * @param element_type The type of elements in the array
     * @param num_elements The number of elements (can be a symbolic expression)
     */
    Array(const IType& element_type, const symbolic::Expression num_elements);

    /**
     * @brief Constructs an Array with explicit storage, alignment, initializer, element type, and size
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     * @param element_type The type of elements in the array
     * @param num_elements The number of elements (can be a symbolic expression)
     */
    Array(
        StorageType storage_type,
        size_t alignment,
        const std::string& initializer,
        const IType& element_type,
        const symbolic::Expression num_elements
    );

    /**
     * @brief Returns the primitive type of the array's elements
     *
     * For nested arrays, this returns the primitive type of the innermost element.
     *
     * @return The primitive type of the elements
     */
    virtual PrimitiveType primitive_type() const override;

    /// @brief Returns TypeID::Array
    virtual TypeID type_id() const override;

    /**
     * @brief Arrays are not symbolic types
     * @return Always returns false
     */
    virtual bool is_symbol() const override;

    bool is_pointer_like() const override { return true; }

    /**
     * @brief Gets the type of elements in this array
     * @return A const reference to the element type
     */
    const IType& element_type() const;

    /**
     * @brief Gets the number of elements in this array
     * @return The number of elements as a symbolic expression
     */
    const symbolic::Expression num_elements() const;

    /// @brief Compares this array with another type for equality
    virtual bool operator==(const IType& other) const override;

    /// @brief Creates a deep copy of this array type
    virtual std::unique_ptr<IType> clone() const override;

    /// @brief Returns a string representation of this array type
    virtual std::string print() const override;
};

} // namespace types
} // namespace sdfg
