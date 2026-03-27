/**
 * @file symbolic.h
 * @brief Symbolic system for integer-based symbol analysis
 *
 * This file defines the symbolic system used throughout sdfglib.
 * Symbolic expressions form the basis of the IR and are used to represent:
 * - Address calculations in memlets
 * - Loop iteration variables and bounds (indvar, bounds)
 * - Conditional expressions in if-else statements (conditions)
 *
 * ## Symbolic System
 *
 * Symbolic expressions are functions of symbols, constants, and other symbolic expressions.
 * The system is thereby constrained to the representation of integer-like symbols:
 * - Every constant integer is a symbolic expression.
 * - Every container declared as an integer is a symbol.
 * - Every pointer declared in the SDFG system is a symbol.
 * - Arithmetic operations (+, -, *, /, %, min, max, abs, pow) on symbolic expressions yield new symbolic expressions.
 *
 * ## Examples
 *
 * Creating symbols and expressions:
 * @code
 * auto x = symbolic::symbol("x");
 * auto y = symbolic::symbol("y");
 * auto expr = symbolic::add(symbolic::mul(x, symbolic::integer(2)), y);  // 2*x + y
 * @endcode
 *
 * Building conditions:
 * @code
 * auto cond = symbolic::And(symbolic::Ge(x, symbolic::zero()), symbolic::Lt(x, symbolic::integer(10)));
 * @endcode
 *
 * @see assumptions.h for information on reasoning about symbol bounds and evolution
 * @see sets.h for integer set operations and disjointness checking
 */

#pragma once

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/nan.h>
#include <symengine/real_double.h>
#include <symengine/simplify.h>
#include <symengine/symbol.h>

#include <unordered_map>
#include "symengine/functions.h"

namespace sdfg {

namespace types {
// forward declaration, because it depends on contents of this file
class IType;
} // namespace types

namespace symbolic {

/**
 * @defgroup symbolic_types Symbolic Type Definitions
 * @brief Core type definitions for symbolic expressions
 * @{
 */

/** @brief Reference-counted pointer to a SymEngine Symbol */
typedef SymEngine::RCP<const SymEngine::Symbol> Symbol;

/** @brief Reference-counted pointer to a SymEngine Number */
typedef SymEngine::RCP<const SymEngine::Number> Number;

/** @brief Reference-counted pointer to a SymEngine Integer constant */
typedef SymEngine::RCP<const SymEngine::Integer> Integer;

/** @brief Reference-counted pointer to a SymEngine Infinity value */
typedef SymEngine::RCP<const SymEngine::Infty> Infty;

/** @brief Reference-counted pointer to a symbolic expression (SymEngine Basic) */
typedef SymEngine::RCP<const SymEngine::Basic> Expression;

/** @brief Reference-counted pointer to a boolean condition expression */
typedef SymEngine::RCP<const SymEngine::Boolean> Condition;

/** @brief Vector of symbolic expressions for multi-dimensional operations */
typedef std::vector<Expression> MultiExpression;

/** @brief Vector of symbols */
typedef std::vector<Symbol> SymbolVec;

/** @brief Ordered set of symbols */
typedef std::set<Symbol, SymEngine::RCPBasicKeyLess> SymbolSet;

/** @brief Map from symbols to expressions */
typedef std::unordered_map<Symbol, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> SymbolMap;

/** @brief Vector of expressions */
typedef std::vector<Symbol> ExpressionVec;

/** @brief Ordered set of expressions */
typedef std::set<Expression, SymEngine::RCPBasicKeyLess> ExpressionSet;

/** @brief Map from expressions to expressions */
typedef std::unordered_map<Expression, Expression, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq> ExpressionMap;

/** @} */ // end of symbolic_types group

/**
 * @defgroup symbolic_creation Symbol and Constant Creation
 * @brief Functions for creating symbols and constant values
 * @{
 */

/**
 * @brief Creates a symbolic variable with the given name
 * @param name The name of the symbol (must not be "null", "NULL", or "nullptr")
 * @return A new Symbol with the specified name
 * @throws InvalidSDFGException if name is a reserved keyword
 *
 * @code
 * auto x = symbolic::symbol("x");
 * auto loop_bound = symbolic::symbol("N");
 * @endcode
 */
Symbol symbol(const std::string& name);

/**
 * @brief Creates an integer constant
 * @param value The integer value
 * @return An Integer constant expression
 *
 * @code
 * auto two = symbolic::integer(2);
 * auto negative = symbolic::integer(-5);
 * @endcode
 */
Integer integer(int64_t value);

/**
 * @brief Creates the integer constant 0
 * @return Zero as an Integer expression
 */
Integer zero();

/**
 * @brief Creates the integer constant 1
 * @return One as an Integer expression
 */
Integer one();

/**
 * @brief Creates a boolean false condition
 * @return False as a Condition
 */
Condition __false__();

/**
 * @brief Creates a boolean true condition
 * @return True as a Condition
 */
Condition __true__();

/**
 * @brief Creates a special null pointer symbol
 * @return A Symbol representing the null pointer
 *
 * In the symbolic system, pointers are interpreted as integers without dereferencing.
 * This function provides a special symbol to represent null pointer values.
 */
Symbol __nullptr__();

/**
 * @brief Checks if a symbol is the null pointer
 * @param symbol The symbol to check
 * @return true if the symbol represents a null pointer
 */
bool is_nullptr(const Symbol symbol);

/**
 * @brief Checks if a symbol represents a pointer
 * @param symbol The symbol to check
 * @return true if the symbol is a pointer (currently equivalent to is_nullptr)
 */
bool is_pointer(const Symbol symbol);

/**
 * @brief Checks if a symbol is an NVIDIA GPU built-in variable
 * @param symbol The symbol to check
 * @return true if the symbol is threadIdx, blockIdx, blockDim, or gridDim in any dimension
 *
 * NVIDIA GPU built-in variables include threadIdx_{x,y,z}, blockIdx_{x,y,z},
 * blockDim_{x,y,z}, and gridDim_{x,y,z}.
 */
bool is_nv(const Symbol symbol);

/**
 * @brief Ceiling division of two expressions
 * @param dividend The dividend expression
 * @param divisor The divisor expression
 * @return An Expression representing ceil(dividend / divisor)
 *
 * This function computes the ceiling of the division of two symbolic expressions.
 * If both expressions are constant, it performs integer arithmetic to compute the result.
 * Otherwise, it constructs a symbolic expression representing the ceiling division.
 *
 * @code
 * auto x = symbolic::symbol("x");
 * auto y = symbolic::symbol("y");
 * auto result = symbolic::divide_ceil(x, y); // Represents ceil(x / y)
 * @endcode
 */
Expression divide_ceil(const Expression dividend, const Expression divisor);

/** @} */ // end of symbolic_creation group

/**
 * @defgroup symbolic_logic Logical Operations
 * @brief Functions for building and evaluating logical conditions
 *
 * These functions construct boolean conditions used in control flow (if-else statements)
 * and loop guards. They follow standard boolean logic semantics.
 * @{
 */

/**
 * @brief Logical AND of two conditions
 * @param lhs Left-hand side condition
 * @param rhs Right-hand side condition
 * @return A condition representing (lhs AND rhs)
 *
 * @code
 * auto cond = symbolic::And(symbolic::Ge(x, zero()), symbolic::Lt(x, integer(10)));
 * // Represents: x >= 0 AND x < 10
 * @endcode
 */
Condition And(const Condition lhs, const Condition rhs);

/**
 * @brief Logical OR of two conditions
 * @param lhs Left-hand side condition
 * @param rhs Right-hand side condition
 * @return A condition representing (lhs OR rhs)
 */
Condition Or(const Condition lhs, const Condition rhs);

/**
 * @brief Logical NOT (negation) of a condition
 * @param expr Condition to negate
 * @return A condition representing NOT expr
 */
Condition Not(const Condition expr);

/**
 * @brief Tests if an expression evaluates to true
 * @param expr Expression to test
 * @return true if expr is the constant true value
 */
bool is_true(const Expression expr);

/**
 * @brief Tests if an expression evaluates to false
 * @param expr Expression to test
 * @return true if expr is the constant false value
 */
bool is_false(const Expression expr);

/** @} */ // end of symbolic_logic group

/**
 * @defgroup symbolic_arithmetic Arithmetic Operations
 * @brief Functions for building arithmetic expressions
 *
 * These functions construct symbolic arithmetic expressions representing integer operations.
 * They are used in address calculations (memlets) and loop bounds.
 *
 * @code
 * // Example: Compute 2*i + j
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * auto expr = symbolic::add(symbolic::mul(symbolic::integer(2), i), j);
 * @endcode
 *
 * @{
 */

/**
 * @brief Addition of two expressions
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Expression representing (lhs + rhs)
 */
Expression add(const Expression lhs, const Expression rhs);

/**
 * @brief Subtraction of two expressions
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Expression representing (lhs - rhs)
 */
Expression sub(const Expression lhs, const Expression rhs);

/**
 * @brief Multiplication of two expressions
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Expression representing (lhs * rhs)
 */
Expression mul(const Expression lhs, const Expression rhs);

/**
 * @brief Integer division of two expressions
 * @param lhs Dividend expression
 * @param rhs Divisor expression
 * @return Expression representing (lhs / rhs) with integer division semantics
 */
Expression div(const Expression lhs, const Expression rhs);

/**
 * @brief Minimum of two expressions
 * @param lhs First expression
 * @param rhs Second expression
 * @return Expression representing min(lhs, rhs)
 */
Expression min(const Expression lhs, const Expression rhs);

/**
 * @brief Maximum of two expressions
 * @param lhs First expression
 * @param rhs Second expression
 * @return Expression representing max(lhs, rhs)
 */
Expression max(const Expression lhs, const Expression rhs);

/**
 * @brief Absolute value of an expression
 * @param expr Expression to take absolute value of
 * @return Expression representing |expr|
 */
Expression abs(const Expression expr);

/**
 * @brief Modulo operation
 * @param lhs Dividend expression
 * @param rhs Divisor expression
 * @return Expression representing (lhs mod rhs)
 */
Expression mod(const Expression lhs, const Expression rhs);

/**
 * @brief Function class for integer modulo operation
 *
 * This class represents the symbolic function for integer modulo.
 */
class IModFunction : public SymEngine::FunctionSymbol {
public:
    IModFunction(const Expression lhs, const Expression rhs) : FunctionSymbol("imod", {lhs, rhs}) {}
};

/**
 * @brief Power/exponentiation operation
 * @param base Base expression
 * @param exp Exponent expression
 * @return Expression representing base^exp
 */
Expression pow(const Expression base, const Expression exp);

/**
 * @brief Zero-extend to 64-bit integer
 * @param expr Expression to zero-extend
 * @return Expression representing the zero-extended value
 */
Expression zext_i64(const Expression expr);

/**
 * @brief Function class for zero-extension to 64-bit integer
 *
 * This class represents a symbolic function that zero-extends its argument to 64-bit.
 */
class ZExtI64Function : public SymEngine::FunctionSymbol {
public:
    explicit ZExtI64Function(const Expression expr) : FunctionSymbol("zext_i64", expr) {}
};

/**
 * @brief Truncate to 32-bit integer
 * @param expr Expression to truncate
 * @return Expression representing the truncated value
 */
Expression trunc_i32(const Expression expr);

/**
 * @brief Function class for truncation to 32-bit integer
 *
 * This class represents a symbolic function that truncates its argument to 32-bit.
 */
class TruncI32Function : public SymEngine::FunctionSymbol {
public:
    explicit TruncI32Function(const Expression expr) : FunctionSymbol("trunc_i32", expr) {}
};

/** @} */ // end of symbolic_arithmetic group

/**
 * @defgroup symbolic_sizeof Size Operations
 * @brief Functions for computing sizes of types and dynamic allocations
 * @{
 */

/**
 * @brief Computes the size of a type
 * @param type The type to compute the size of
 * @return Expression representing sizeof(type)
 */
Expression size_of_type(const types::IType& type);

/**
 * @brief Function class representing sizeof for a type
 *
 * This class represents a symbolic sizeof operation on a type, used for
 * static size computations in the type system.
 */
class SizeOfTypeFunction : public SymEngine::FunctionSymbol {
private:
    const types::IType& type_;

public:
    explicit SizeOfTypeFunction(const types::IType& type)
        : FunctionSymbol("sizeof", SymEngine::vec_basic{}), type_(type) {}

    const types::IType& get_type() const { return type_; }
};

/**
 * @brief Computes the dynamic size of a memory region
 * @param symbol Symbol representing a memory allocation
 * @return Expression representing the dynamic size of the allocation
 *
 * This is used for dynamically-sized allocations where the size is not
 * known statically.
 */
Expression dynamic_sizeof(const Symbol symbol);

/**
 * @brief Function class for dynamic sizeof operation
 */
class DynamicSizeOfFunction : public SymEngine::FunctionSymbol {
public:
    explicit DynamicSizeOfFunction(const Symbol symbol) : FunctionSymbol("dynamic_sizeof", symbol) {}
};

/**
 * @brief Computes the usable size of a malloc allocation
 * @param symbol Symbol representing a malloc'd pointer
 * @return Expression representing malloc_usable_size(symbol)
 *
 * This represents the actual usable size of a memory allocation, which may
 * be larger than the requested size due to allocator alignment.
 */
Expression malloc_usable_size(const Symbol symbol);

/**
 * @brief Function class for malloc_usable_size operation
 */
class MallocUsableSizeFunction : public SymEngine::FunctionSymbol {
public:
    explicit MallocUsableSizeFunction(const Symbol symbol) : FunctionSymbol("malloc_usable_size", symbol) {}
};

/** @} */ // end of symbolic_sizeof group

/**
 * @defgroup symbolic_comparison Comparison Operations
 * @brief Functions for building comparison conditions
 *
 * These functions construct relational comparison expressions that evaluate to boolean conditions.
 * Used extensively in loop bounds and conditional statements.
 * @{
 */

/**
 * @brief Equality comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs == rhs)
 */
Condition Eq(const Expression lhs, const Expression rhs);

/**
 * @brief Inequality comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs != rhs)
 */
Condition Ne(const Expression lhs, const Expression rhs);

/**
 * @brief Less-than comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs < rhs)
 */
Condition Lt(const Expression lhs, const Expression rhs);

/**
 * @brief Greater-than comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs > rhs)
 */
Condition Gt(const Expression lhs, const Expression rhs);

/**
 * @brief Less-than-or-equal comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs <= rhs)
 */
Condition Le(const Expression lhs, const Expression rhs);

/**
 * @brief Greater-than-or-equal comparison
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @return Condition representing (lhs >= rhs)
 */
Condition Ge(const Expression lhs, const Expression rhs);

/** @} */ // end of symbolic_comparison group

/**
 * @defgroup symbolic_manipulation Expression Manipulation
 * @brief Functions for transforming and analyzing expressions
 *
 * These functions provide utilities for expanding, simplifying, and analyzing symbolic expressions.
 * They are essential for expression optimization and analysis.
 * @{
 */

/**
 * @brief Expands an expression algebraically
 * @param expr Expression to expand
 * @return Expanded form of the expression
 *
 * Performs algebraic expansion, such as distributing multiplications over additions.
 */
Expression expand(const Expression expr);

/**
 * @brief Simplifies an expression
 * @param expr Expression to simplify
 * @return Simplified form of the expression
 *
 * Applies simplification rules to reduce the expression to a canonical form.
 */
Expression simplify(const Expression expr);

/**
 * @brief Tests structural equality of two expressions
 * @param lhs First expression
 * @param rhs Second expression
 * @return true if the expressions are structurally equal
 */
bool eq(const Expression lhs, const Expression rhs);

/**
 * @brief Tests equality of expressions with null-safety
 * @param lhs First expression (can be null)
 * @param rhs Second expression (can be null)
 * @return true if expressions are equal; both null is considered equal
 *
 * This is a null-safe version of eq() where both inputs being null is also considered equal.
 */
bool null_safe_eq(const Expression lhs, const Expression rhs);

/**
 * @brief Tests if an expression uses a specific symbol
 * @param expr Expression to search
 * @param sym Symbol to look for
 * @return true if expr contains sym
 */
bool uses(const Expression expr, const Symbol sym);

/**
 * @brief Tests if an expression uses a symbol with a specific name
 * @param expr Expression to search
 * @param name Name of the symbol to look for
 * @return true if expr contains a symbol with the given name
 */
bool uses(const Expression expr, const std::string& name);

/**
 * @brief Extracts all symbols (atoms) from an expression
 * @param expr Expression to analyze
 * @return Set of all symbols appearing in the expression
 */
SymbolSet atoms(const Expression expr);

/**
 * @brief Checks if an expression contains a specific SymEngine type
 * @tparam T SymEngine type to search for
 * @param expr Expression to search
 * @return true if expr contains an instance of type T
 */
template<typename T>
inline bool has(const Expression expr) {
    for (auto& atom : SymEngine::atoms<T>(*expr)) {
        if (SymEngine::is_a<T>(*atom)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Finds all instances of a specific SymEngine type in an expression
 * @tparam T SymEngine type to search for
 * @param expr Expression to search
 * @return Set of all sub-expressions of type T
 */
template<typename T>
inline ExpressionSet find(const Expression expr) {
    ExpressionSet res;
    for (auto& atom : SymEngine::atoms<T>(*expr)) {
        if (SymEngine::is_a<T>(*atom)) {
            res.insert(atom);
        }
    }
    return res;
}

/**
 * @brief Checks if an expression contains a dynamic_sizeof operation
 * @param expr Expression to check
 * @return true if expr contains dynamic_sizeof
 */
bool has_dynamic_sizeof(const Expression expr);

/**
 * @brief Extracts all multiplication operations from an expression
 * @param expr Expression to analyze
 * @return Set of multiplication sub-expressions
 */
ExpressionSet muls(const Expression expr);

/**
 * @brief Substitutes a sub-expression with another expression
 * @param expr Expression to perform substitution in
 * @param old_expr Sub-expression to replace
 * @param new_expr Expression to substitute
 * @return New expression with substitution applied
 *
 * @code
 * auto x = symbolic::symbol("x");
 * auto expr = symbolic::add(x, symbolic::integer(1));  // x + 1
 * auto result = symbolic::subs(expr, x, symbolic::integer(5));  // 5 + 1 = 6
 * @endcode
 */
Expression subs(const Expression expr, const Expression old_expr, const Expression new_expr);

/**
 * @brief Substitutes a sub-expression in a condition
 * @param expr Condition to perform substitution in
 * @param old_expr Sub-expression to replace
 * @param new_expr Expression to substitute
 * @return New condition with substitution applied
 */
Condition subs(const Condition expr, const Expression old_expr, const Expression new_expr);

/**
 * @brief Computes the inverse of an expression with respect to a symbol
 * @param expr Expression to invert
 * @param symbol Symbol to solve for
 * @return Inverse expression, or null if inverse doesn't exist
 *
 * For an expression y = f(x), computes x = f^(-1)(y).
 */
Expression inverse(const Expression expr, const Symbol symbol);

/** @} */ // end of symbolic_manipulation group

/**
 * @defgroup symbolic_gpu GPU Built-in Symbols
 * @brief NVIDIA CUDA built-in variable symbols
 *
 * These functions provide access to NVIDIA GPU thread and block indexing symbols.
 * They are used when generating GPU code to reference thread/block indices and dimensions.
 * @{
 */

/** @brief Thread index in X dimension */
Symbol threadIdx_x();

/** @brief Thread index in Y dimension */
Symbol threadIdx_y();

/** @brief Thread index in Z dimension */
Symbol threadIdx_z();

/** @brief Block dimension (threads per block) in X */
Symbol blockDim_x();

/** @brief Block dimension (threads per block) in Y */
Symbol blockDim_y();

/** @brief Block dimension (threads per block) in Z */
Symbol blockDim_z();

/** @brief Block index in X dimension */
Symbol blockIdx_x();

/** @brief Block index in Y dimension */
Symbol blockIdx_y();

/** @brief Block index in Z dimension */
Symbol blockIdx_z();

/** @brief Grid dimension (blocks per grid) in X */
Symbol gridDim_x();

/** @brief Grid dimension (blocks per grid) in Y */
Symbol gridDim_y();

/** @brief Grid dimension (blocks per grid) in Z */
Symbol gridDim_z();

/** @} */ // end of symbolic_gpu group

/**
 * @defgroup symbolic_parsing Expression Parsing
 * @brief Functions for parsing string representations
 * @{
 */

/**
 * @brief Parses a string into a symbolic expression
 * @param expr_str String representation of an expression
 * @return Parsed symbolic expression
 *
 * Uses SymEngine's parser to convert string representations into symbolic expressions.
 *
 * @code
 * auto expr = symbolic::parse("2*x + y - 3");
 * @endcode
 */
Expression parse(const std::string& expr_str);

/** @} */ // end of symbolic_parsing group

} // namespace symbolic
} // namespace sdfg
