/**
 * @file utils.h
 * @brief Utility functions for working with types
 *
 * This file provides various utility functions for type manipulation, inspection,
 * and analysis including type inference, size calculation, and type decomposition.
 */

#pragma once

#include <cassert>
#include <memory>
#include <unordered_map>
#include <vector>

#include "sdfg/function.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {

namespace data_flow {

/// @brief Type alias for a subset specification (array of symbolic expressions)
typedef std::vector<symbolic::Expression> Subset;

} // namespace data_flow

namespace types {

/**
 * @brief Infers the type resulting from accessing a subset of data
 *
 * Given a type and a subset (index expressions), this function determines what
 * type results from that access. For example, accessing an element of an array
 * returns the element type, while accessing a full array returns the array type.
 *
 * @param function The function context
 * @param type The type being accessed
 * @param subset The subset/index expressions defining the access
 * @return A const reference to the inferred type
 */
std::unique_ptr<types::IType>
infer_type(const sdfg::Function& function, const types::IType& type, const data_flow::Subset& subset);

/**
 * @brief Reconstructs an array type with a new inner element type
 *
 * This function is used to rebuild array types after transformations. It reconstructs
 * the array type hierarchy to a specified depth with a new innermost element type.
 *
 * @param type The original type
 * @param depth The depth to which to reconstruct the array type
 * @param inner_type The new innermost element type
 * @return A unique pointer to the reconstructed type
 */
std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth, const types::IType& inner_type);

/// @brief Special value indicating to follow only an outermost pointer when peeling
inline constexpr int PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR = -100;

/**
 * @brief Returns the innermost element type of a (potentially multi-dimensional) array
 *
 * This function recursively unwraps array and pointer types to find the innermost
 * element type. The follow_ptr parameter controls how pointers are handled during
 * unwrapping.
 *
 * @param type The type to unwrap
 * @param follow_ptr Controls pointer following behavior:
 *        - PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR (default): Only follow an outermost pointer
 *        - Positive count: Follow that many pointers
 *        - -1: Follow infinite pointers
 * @return A const reference to the innermost element type
 */
const IType& peel_to_innermost_element(const IType& type, int follow_ptr = PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR);

/**
 * @brief Returns the size in bytes of one contiguous element
 *
 * This computes the size of a single element according to peel_to_innermost_element.
 * This is useful for calculating strides and offsets in array indexing.
 *
 * @param type The type to query
 * @param allow_comp_time_eval Whether to emit compiler-time sizeof expressions if static size is unknown (default:
 * true)
 * @return The element size as a symbolic expression (in bytes)
 */
symbolic::Expression get_contiguous_element_size(const types::IType& type, bool allow_comp_time_eval = true);

/**
 * @brief Returns the size of a type in bytes
 *
 * Computes the total size of the given type. For arrays, this includes all elements.
 * For structures, this includes all members with padding.
 *
 * @param type The type to query
 * @param allow_comp_time_eval Whether to emit compiler-time sizeof expressions if static size is unknown (default:
 * true)
 * @return The type size as a symbolic expression (in bytes), or empty RCP if unknown
 */
symbolic::Expression get_type_size(const types::IType& type, bool allow_comp_time_eval = true);

/**
 * @brief Returns the next element type inside an array/pointer/reference
 *
 * This function unwraps one level of array, pointer, or reference type to get
 * the type of the element it contains or points to.
 *
 * @param type The type to peel
 * @return A pointer to the next element type, or nullptr if there is none
 */
const types::IType* peel_to_next_element(const types::IType& type);

/**
 * @brief Checks if a type is contiguous in memory
 *
 * This function determines if the given type represents a contiguous memory layout.
 * Non-contiguous types include types with nested pointers.
 * Outermost pointers are allowed as they can still point to contiguous memory.
 * @param type The type to check
 * @param sdfg The StructuredSDFG context, to access structure definitions if needed
 * @return True if the type is contiguous, false otherwise
 */
bool is_contiguous_type(const types::IType& type, StructuredSDFG& sdfg);

/**
 * Return a good primitive type that can hold the expressions maximum value.
 * Current limited impl: will evaluate the expression. If result is immediate and fits into Int32, then that,
 * otherwise falls back to Int64
 * @return For now Int32 or Int64 if not sure
 */
PrimitiveType get_primitive_type_to_hold_upper_bound(const symbolic::Expression& expr);

/**
 * Return a good primitive type that can hold the expressions minimum value.
 * Current limited impl: will evaluate the expression. If result is immediate and fits into Int32, then that,
 * otherwise falls back to Int64
 * @return For now Int32 or Int64 if not sure
 */
PrimitiveType get_primitive_type_to_hold_lower_bound(const symbolic::Expression& expr);

/**
 * Return the primitive type that can hold the contents of both a and b without any risk of loss
 * (respecting C-style signed-overflow is undefined and need not be considered. but unsigned overflow does)
 * @return
 */
PrimitiveType get_primitive_type_to_hold(PrimitiveType a, PrimitiveType b);

} // namespace types
} // namespace sdfg
