/**
 * @file tasklet.h
 * @brief Simple computational operations in the dataflow graph
 *
 * Tasklets represent primitive computational operations like arithmetic, logic,
 * and type conversions. They are the fundamental building blocks for computation
 * in the SDFG dataflow graph.
 *
 * ## Key Concepts
 *
 * ### Tasklet Operations
 * Tasklets support various operation types:
 * - **assign**: Direct assignment/copy
 * - **Floating-point**: fp_add, fp_mul, fp_div, fp_neg, fp_fma, etc.
 * - **Integer**: int_add, int_mul, int_and, int_or, int_shl, etc.
 * - **Comparisons**: fp_oeq, int_eq, int_slt, etc.
 *
 * ### Type Conversions
 * The assign operation can represent type conversions when input and output
 * types differ:
 * - **zext**: Zero extend (unsigned to larger)
 * - **sext**: Sign extend (signed to larger)
 * - **trunc**: Truncate (larger to smaller)
 * - **fptoui/fptosi**: Float to integer
 * - **uitofp/sitofp**: Integer to float
 * - **fpext/fptrunc**: Float precision conversion
 *
 * ### Operation Properties
 * Each operation has:
 * - **Arity**: Number of inputs (1, 2, or 3)
 * - **Type**: Integer, floating-point, or generic
 * - **Signedness**: For integer operations
 *
 * ## Example Usage
 *
 * Creating a simple addition tasklet:
 * @code
 * auto& tasklet = builder.add_tasklet(
 *     state,
 *     TaskletCode::fp_add,  // operation
 *     "_out",               // output connector
 *     {"_in1", "_in2"}      // input connectors
 * );
 * @endcode
 *
 * Creating an assignment/conversion:
 * @code
 * auto& tasklet = builder.add_tasklet(
 *     state,
 *     TaskletCode::assign,
 *     "_out",
 *     {"_in"}
 * );
 * // If input and output types differ, this becomes a type conversion
 * @endcode
 *
 * Querying tasklet properties:
 * @code
 * TaskletCode code = tasklet.code();
 * bool is_cast = tasklet.is_cast(function);
 * bool is_trivial = tasklet.is_trivial(function);
 * size_t num_inputs = arity(code);
 * @endcode
 *
 * @see CodeNode for the base class
 * @see LibraryNode for complex operations
 * @see DataFlowNode for node interface
 */

#pragma once

#include "sdfg/data_flow/code_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class FunctionBuilder;
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

/**
 * @enum TaskletCode
 * @brief Operation codes for tasklet computations
 *
 * Defines all primitive operations that tasklets can perform, including
 * arithmetic, logical, comparison, and type conversion operations.
 */
enum TaskletCode {
    assign, ///< Assignment/copy operation (can be cast if types differ)
    // Floating-point-specific operations
    // Operations
    fp_neg, ///< Floating-point negation
    fp_add, ///< Floating-point addition
    fp_sub, ///< Floating-point subtraction
    fp_mul, ///< Floating-point multiplication
    fp_div, ///< Floating-point division
    fp_rem, ///< Floating-point remainder (fmod semantics)
    fp_fma, ///< Floating-point fused multiply-add
    // Comparisions
    fp_oeq, ///< Floating-point ordered equal
    fp_one, ///< Floating-point ordered not equal
    fp_oge, ///< Floating-point ordered greater or equal
    fp_ogt, ///< Floating-point ordered greater than
    fp_ole, ///< Floating-point ordered less or equal
    fp_olt, ///< Floating-point ordered less than
    fp_ord, ///< Floating-point ordered (neither operand is NaN)
    fp_ueq, ///< Floating-point unordered equal
    fp_une, ///< Floating-point unordered not equal
    fp_ugt, ///< Floating-point unordered greater than
    fp_uge, ///< Floating-point unordered greater or equal
    fp_ult, ///< Floating-point unordered less than
    fp_ule, ///< Floating-point unordered less or equal
    fp_uno, ///< Floating-point unordered (at least one operand is NaN)
    // Integer-specific operations
    // Operations
    int_add, ///< Integer addition
    int_sub, ///< Integer subtraction
    int_mul, ///< Integer multiplication
    int_sdiv, ///< Signed integer division
    int_srem, ///< Signed integer remainder
    int_udiv, ///< Unsigned integer division
    int_urem, ///< Unsigned integer remainder
    int_and, ///< Bitwise AND
    int_or, ///< Bitwise OR
    int_xor, ///< Bitwise XOR
    int_shl, ///< Shift left
    int_ashr, ///< Arithmetic shift right (sign-extend)
    int_lshr, ///< Logical shift right (zero-extend)
    int_smin, ///< Signed minimum
    int_smax, ///< Signed maximum
    int_scmp, ///< Signed comparison (-1, 0, 1)
    int_umin, ///< Unsigned minimum
    int_umax, ///< Unsigned maximum
    int_ucmp, ///< Unsigned comparison (-1, 0, 1)
    // Comparisions
    int_eq, ///< Integer equal
    int_ne, ///< Integer not equal
    int_sge, ///< Signed greater or equal
    int_sgt, ///< Signed greater than
    int_sle, ///< Signed less or equal
    int_slt, ///< Signed less than
    int_uge, ///< Unsigned greater or equal
    int_ugt, ///< Unsigned greater than
    int_ule, ///< Unsigned less or equal
    int_ult, ///< Unsigned less than
    int_abs ///< Integer absolute value
};

/**
 * @brief Get the number of inputs for a tasklet operation
 * @param c TaskletCode operation
 * @return Number of inputs required (arity)
 * @throws InvalidSDFGException if code is invalid
 */
constexpr size_t arity(TaskletCode c) {
    switch (c) {
        case TaskletCode::assign:
        case TaskletCode::int_abs:
            return 1;
        // Integer Relational Ops
        case TaskletCode::int_add:
        case TaskletCode::int_sub:
        case TaskletCode::int_mul:
        case TaskletCode::int_sdiv:
        case TaskletCode::int_srem:
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_and:
        case TaskletCode::int_or:
        case TaskletCode::int_xor:
        case TaskletCode::int_shl:
        case TaskletCode::int_ashr:
        case TaskletCode::int_lshr:
        case TaskletCode::int_smin:
        case TaskletCode::int_smax:
        case TaskletCode::int_umin:
        case TaskletCode::int_scmp:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
            return 2;
        // Comparisions
        case TaskletCode::int_eq:
        case TaskletCode::int_ne:
        case TaskletCode::int_sge:
        case TaskletCode::int_sgt:
        case TaskletCode::int_sle:
        case TaskletCode::int_slt:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return 2;
        // Floating Point
        case TaskletCode::fp_neg:
            return 1;
        case TaskletCode::fp_add:
        case TaskletCode::fp_sub:
        case TaskletCode::fp_mul:
        case TaskletCode::fp_div:
        case TaskletCode::fp_rem:
            return 2;
        // Comparisions
        case TaskletCode::fp_oeq:
        case TaskletCode::fp_one:
        case TaskletCode::fp_oge:
        case TaskletCode::fp_ogt:
        case TaskletCode::fp_ole:
        case TaskletCode::fp_olt:
        case TaskletCode::fp_ord:
        case TaskletCode::fp_ueq:
        case TaskletCode::fp_une:
        case TaskletCode::fp_ugt:
        case TaskletCode::fp_uge:
        case TaskletCode::fp_ult:
        case TaskletCode::fp_ule:
        case TaskletCode::fp_uno:
            return 2;
        case TaskletCode::fp_fma:
            return 3;
    };
    throw InvalidSDFGException("Invalid tasklet code");
};

/**
 * @brief Check if a tasklet operation is unsigned
 * @param c TaskletCode operation
 * @return True if the operation treats operands as unsigned
 */
constexpr bool is_unsigned(TaskletCode c) {
    switch (c) {
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_lshr:
        case TaskletCode::int_umin:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return true;
        default:
            return false;
    }
};

/**
 * @brief Check if a tasklet operation is an integer operation
 * @param c TaskletCode operation
 * @return True if the operation works on integer types
 */
constexpr bool is_integer(TaskletCode c) {
    switch (c) {
        // Operations
        case TaskletCode::int_add:
        case TaskletCode::int_sub:
        case TaskletCode::int_mul:
        case TaskletCode::int_sdiv:
        case TaskletCode::int_srem:
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_and:
        case TaskletCode::int_or:
        case TaskletCode::int_xor:
        case TaskletCode::int_shl:
        case TaskletCode::int_ashr:
        case TaskletCode::int_lshr:
        case TaskletCode::int_smin:
        case TaskletCode::int_smax:
        case TaskletCode::int_scmp:
        case TaskletCode::int_umin:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
        case TaskletCode::int_abs:
        // Comparisions
        case TaskletCode::int_eq:
        case TaskletCode::int_ne:
        case TaskletCode::int_sge:
        case TaskletCode::int_sgt:
        case TaskletCode::int_sle:
        case TaskletCode::int_slt:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a tasklet operation is a floating-point operation
 * @param c TaskletCode operation
 * @return True if the operation works on floating-point types
 */
constexpr bool is_floating_point(TaskletCode c) {
    switch (c) {
        // Operations
        case TaskletCode::fp_neg:
        case TaskletCode::fp_add:
        case TaskletCode::fp_sub:
        case TaskletCode::fp_mul:
        case TaskletCode::fp_div:
        case TaskletCode::fp_rem:
        case TaskletCode::fp_fma:
        // Comparisions
        case TaskletCode::fp_oeq:
        case TaskletCode::fp_one:
        case TaskletCode::fp_oge:
        case TaskletCode::fp_ogt:
        case TaskletCode::fp_ole:
        case TaskletCode::fp_olt:
        case TaskletCode::fp_ord:
        case TaskletCode::fp_ueq:
        case TaskletCode::fp_une:
        case TaskletCode::fp_ugt:
        case TaskletCode::fp_uge:
        case TaskletCode::fp_ult:
        case TaskletCode::fp_ule:
        case TaskletCode::fp_uno:
            return true;
        default:
            return false;
    }
};

/**
 * @class Tasklet
 * @brief Simple computational operation in the dataflow graph
 *
 * Tasklets represent primitive operations like arithmetic, comparisons, and
 * type conversions. They are lightweight code nodes with a single operation
 * code and fixed input/output structure.
 *
 * Key features:
 * - Single operation per tasklet
 * - Type-aware (integer, floating-point, or generic)
 * - Automatic type conversion detection
 * - LLVM-like operation semantics
 */
class Tasklet : public CodeNode {
    friend class sdfg::builder::FunctionBuilder;
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    TaskletCode code_; ///< Operation code for this tasklet

    /**
     * @brief Constructor for tasklets
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this tasklet
     * @param vertex Graph vertex for this tasklet
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param output Output connector name
     * @param inputs Input connector names
     */
    Tasklet(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs
    );

public:
    Tasklet(const Tasklet& data_node) = delete;
    Tasklet& operator=(const Tasklet&) = delete;

    /**
     * @brief Validate the tasklet
     * @param function Function context for validation
     * @throws InvalidSDFGException if validation fails
     */
    void validate(const Function& function) const override;

    [[nodiscard]] bool side_effect() const override { return false; }

    /**
     * @brief Get the operation code
     * @return TaskletCode for this tasklet
     */
    TaskletCode code() const;

    /**
     * @brief Check if this is an assignment operation
     * @return True if the code is TaskletCode::assign
     */
    bool is_assign() const;

    /**
     * @brief Check if this is a trivial (no-op) assignment
     * @param function Function context for type checking
     * @return True if assign with identical input/output types
     */
    bool is_trivial(const Function& function) const;

    /**
     * @brief Check if this is a type cast
     * @param function Function context for type checking
     * @return True if assign with different input/output types
     */
    bool is_cast(const Function& function) const;

    /**
     * @brief Check if this is a zero-extend cast
     * @param function Function context for type checking
     * @return True if unsigned to larger unsigned integer
     */
    bool is_zext(const Function& function) const;

    /**
     * @brief Check if this is a sign-extend cast
     * @param function Function context for type checking
     * @return True if signed to larger signed integer
     */
    bool is_sext(const Function& function) const;

    /**
     * @brief Check if this is a truncate cast
     * @param function Function context for type checking
     * @return True if larger to smaller integer
     */
    bool is_trunc(const Function& function) const;

    /**
     * @brief Check if this is a float-to-unsigned-int cast
     * @param function Function context for type checking
     * @return True if float/double to unsigned integer
     */
    bool is_fptoui(const Function& function) const;

    /**
     * @brief Check if this is a float-to-signed-int cast
     * @param function Function context for type checking
     * @return True if float/double to signed integer
     */
    bool is_fptosi(const Function& function) const;

    /**
     * @brief Check if this is an unsigned-int-to-float cast
     * @param function Function context for type checking
     * @return True if unsigned integer to float/double
     */
    bool is_uitofp(const Function& function) const;

    /**
     * @brief Check if this is a signed-int-to-float cast
     * @param function Function context for type checking
     * @return True if signed integer to float/double
     */
    bool is_sitofp(const Function& function) const;

    /**
     * @brief Check if this is a float precision extension cast
     * @param function Function context for type checking
     * @return True if float to double
     */
    bool is_fpext(const Function& function) const;

    /**
     * @brief Check if this is a float precision truncation cast
     * @param function Function context for type checking
     * @return True if double to float
     */
    bool is_fptrunc(const Function& function) const;

    /**
     * @brief Get the output connector name (single output)
     * @return Name of the output connector
     */
    const std::string& output() const;

    /**
     * @brief Clone this tasklet for graph transformations
     * @param element_id New element identifier
     * @param vertex New graph vertex
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned tasklet
     */
    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    /**
     * @brief Replace symbolic expressions in this tasklet
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    EdgeRemoveOption can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const override;
};
} // namespace data_flow
} // namespace sdfg
