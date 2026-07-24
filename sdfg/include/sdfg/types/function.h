/**
 * @file function.h
 * @brief Function type representation
 *
 * This file defines the Function class, which represents function types including
 * parameter types, return type, and variadic argument support.
 */

#pragma once

#include "sdfg/types/type.h"

namespace sdfg {

namespace types {

/**
 * @class Function
 * @brief Represents a function type
 *
 * The Function class represents a function signature, including:
 * - Parameter types (in order)
 * - Return type
 * - Whether the function accepts variadic arguments (like C's ...)
 *
 * Function types are used to represent function pointers, function declarations,
 * and callable objects in the SDFG type system.
 *
 * @note This represents the type of a function, not a function definition or call.
 */
class Function : public IType {
private:
    std::vector<std::unique_ptr<IType>> params_; ///< Parameter types in order
    std::unique_ptr<IType> return_type_; ///< Return type of the function
    bool is_var_arg_; ///< Whether function accepts variadic arguments

public:
    /**
     * @brief Constructs a Function with the specified return type
     * @param return_type The return type of the function
     * @param is_var_arg Whether the function accepts variadic arguments (default: false)
     */
    Function(const IType& return_type, bool is_var_arg = false);

    /**
     * @brief Constructs a Function with explicit storage, alignment, initializer, and return type
     * @param storage_type The storage location and management
     * @param alignment Memory alignment in bytes
     * @param initializer Optional initializer expression
     * @param return_type The return type of the function
     * @param is_var_arg Whether the function accepts variadic arguments (default: false)
     */
    Function(
        StorageType storage_type,
        size_t alignment,
        const std::string& initializer,
        const IType& return_type,
        bool is_var_arg = false
    );

    /**
     * @brief Returns the primitive type of the return type
     * @return The primitive type of the function's return type
     */
    virtual PrimitiveType primitive_type() const override;

    /// @brief Returns TypeID::Function
    virtual TypeID type_id() const override;

    /**
     * @brief Functions are not symbolic types
     * @return Always returns false
     */
    virtual bool is_symbol() const override;

    bool is_pointer_like() const override { return true; }

    /**
     * @brief Gets the number of parameters
     * @return The number of parameters in the function signature
     */
    size_t num_params() const;

    /**
     * @brief Gets the type of a parameter at the specified index
     * @param index The zero-based index of the parameter
     * @return A const reference to the parameter's type
     */
    const IType& param_type(symbolic::Integer index) const;

    /**
     * @brief Adds a parameter to the function signature
     *
     * Parameters are added in order. This method should be called for each
     * parameter in the function signature.
     *
     * @param param The type of the parameter to add
     */
    void add_param(const IType& param);

    /**
     * @brief Gets the return type of the function
     * @return A const reference to the return type
     */
    const IType& return_type() const;

    /**
     * @brief Checks if the function accepts variadic arguments
     * @return true if the function is variadic (like C's ...), false otherwise
     */
    bool is_var_arg() const;

    /// @brief Compares this function type with another type for equality
    virtual bool operator==(const IType& other) const override;

    /// @brief Creates a deep copy of this function type
    virtual std::unique_ptr<IType> clone() const override;

    /// @brief Returns a string representation of this function type
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
