/**
 * @file structure.h
 * @brief Structure and record type representations
 *
 * This file defines classes for representing structured types (similar to C structs)
 * and their definitions, including member types and layout information.
 */

#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

class Scalar;

namespace builder {
class FunctionBuilder;
} // namespace builder

/**
 * @class Structure
 * @brief Represents a structure type (similar to C struct)
 *
 * Structure types represent composite data types with named members of different types.
 * This is analogous to C/C++ structs. The Structure class holds a reference to a
 * structure type by name, while the actual member layout is defined in StructureDefinition.
 *
 * Structures can also represent SIMD vector types when configured appropriately.
 */
class Structure : public IType {
private:
    std::string name_; ///< Name of the structure type

public:
    /**
     * @brief Constructs a Structure with the specified name
     * @param name The name of the structure type
     */
    Structure(const std::string& name);

    /**
     * @brief Constructs a Structure with explicit storage, alignment, initializer, and name
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     * @param name The name of the structure type
     */
    Structure(StorageType storage_type, size_t alignment, const std::string& initializer, const std::string& name);

    /**
     * @brief Returns the primitive type (implementation-defined for structures)
     * @return The primitive type
     */
    virtual PrimitiveType primitive_type() const override;

    /// @brief Returns TypeID::Structure
    virtual TypeID type_id() const override;

    /**
     * @brief Structures are not symbolic types
     * @return Always returns false
     */
    virtual bool is_symbol() const override;

    /**
     * @brief Gets the name of this structure type
     * @return A const reference to the structure name
     */
    const std::string& name() const;

    /// @brief Compares this structure with another type for equality
    virtual bool operator==(const IType& other) const override;

    /// @brief Creates a deep copy of this structure type
    virtual std::unique_ptr<IType> clone() const override;

    /// @brief Returns a string representation of this structure type
    virtual std::string print() const override;

    /**
     * @brief Checks if this structure behaves like a pointer
     *
     * Some structures may have pointer-like semantics in certain contexts.
     *
     * @return true if this structure is pointer-like, false otherwise
     */
    bool is_pointer_like() const override;
};

/**
 * @class StructureDefinition
 * @brief Defines the layout and members of a structure type
 *
 * StructureDefinition contains the complete definition of a structure, including
 * all member types in order and layout information (packed vs. unpacked).
 *
 * This class also supports creating and identifying SIMD vector types, which are
 * represented as structures with homogeneous scalar members.
 */
class StructureDefinition {
private:
    std::string name_; ///< Name of the structure
    bool is_packed_; ///< Whether the structure is packed (no padding)
    std::vector<std::unique_ptr<IType>> members_; ///< Member types in declaration order

public:
    /**
     * @brief Constructs a StructureDefinition with the specified name and packing
     * @param name The name of the structure
     * @param is_packed Whether the structure should be packed (no padding between members)
     */
    StructureDefinition(const std::string& name, bool is_packed);

    /**
     * @brief Creates a deep copy of this structure definition
     * @return A unique pointer to a new StructureDefinition with the same properties
     */
    std::unique_ptr<StructureDefinition> clone() const;

    /**
     * @brief Gets the name of this structure
     * @return A const reference to the structure name
     */
    const std::string& name() const;

    /**
     * @brief Checks if this structure is packed
     * @return true if the structure is packed, false otherwise
     */
    bool is_packed() const;

    /**
     * @brief Gets the number of members in this structure
     * @return The number of members
     */
    size_t num_members() const;

    /**
     * @brief Gets the type of a member at the specified index
     * @param index The zero-based index of the member
     * @return A const reference to the member's type
     */
    const IType& member_type(symbolic::Integer index) const;

    /**
     * @brief Adds a member to this structure
     * @param member_type The type of the member to add
     */
    void add_member(const IType& member_type);

    /**
     * @brief Checks if this structure represents a SIMD vector type
     *
     * A structure is considered a vector if all its members are scalar types
     * of the same primitive type.
     *
     * @return true if this is a vector type, false otherwise
     */
    bool is_vector() const;

    /**
     * @brief Gets the element type of a vector structure
     * @return A const reference to the scalar element type
     * @pre is_vector() must return true
     */
    const Scalar& vector_element_type() const;

    /**
     * @brief Gets the size of a vector structure
     * @return The number of elements in the vector
     * @pre is_vector() must return true
     */
    const size_t vector_size() const;

    /**
     * @brief Creates or retrieves a vector type structure
     * @param builder The function builder context
     * @param element_type The scalar type of vector elements
     * @param vector_size The number of elements in the vector
     * @return A const reference to the Structure representing the vector type
     */
    static const Structure&
    create_vector_type(const builder::FunctionBuilder& builder, const Scalar& element_type, size_t vector_size);
};

} // namespace types
} // namespace sdfg
