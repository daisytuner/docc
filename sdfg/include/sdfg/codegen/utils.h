#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>

#include "sdfg/types/type.h"

namespace sdfg {

class Function;

namespace data_flow {
class Tasklet;
} // namespace data_flow

namespace codegen {

class PrettyPrinter {
public:
    // Constructor
    PrettyPrinter(int indent = 0, bool frozen = false);
    PrettyPrinter(std::ostream& stream, int indent = 0, bool frozen = false);

    // Set the indentation level
    void setIndent(int indent);

    int indent() const;

    int changeIndent(int delta);

    // Get the underlying string
    std::string str() const;

    // Clear the stringstream content
    void clear();

    // Overload the insertion operator
    template<typename T>
    PrettyPrinter& operator<<(const T& value) {
        if (frozen_) {
            throw std::runtime_error("PrettyPrinter is frozen");
        }
        applyIndent();
        stream << value;
        return *this;
    }

    // Overload for manipulators (like std::endl)
    PrettyPrinter& operator<<(std::ostream& (*manip)(std::ostream&) );

private:
    std::unique_ptr<std::stringstream> owned_stream;
    std::ostream& stream;
    int indentSize;
    bool isNewLine = true;
    bool frozen_;

    // Apply indentation only at the beginning of a new line
    void applyIndent();
};

class Reference : public types::IType {
private:
    std::unique_ptr<types::IType> reference_;

public:
    Reference(const types::IType& reference_);

    Reference(
        types::StorageType storage_type, size_t alignment, const std::string& initializer, const types::IType& reference_
    );

    std::unique_ptr<types::IType> clone() const override;

    virtual types::TypeID type_id() const override;

    types::PrimitiveType primitive_type() const override;

    bool is_symbol() const override;

    bool is_pointer_like() const override { return true; }

    const types::IType& reference_type() const;

    bool operator==(const types::IType& other) const override;

    std::string print() const override;
};

/**
 * @brief Maps a complex primitive type to its generated 2-component vector type name.
 *
 * Uses dedicated `__daisy_type_complex_*` structs with `.x` (real) and `.y` (imaginary)
 * members, defined by complex_support_preamble(). The reserved name prefix avoids collisions
 * with a toolchain's native float2/double2 definitions.
 *
 * @param prim_type A complex primitive type (CHalf, CBFloat, CFloat, CDouble, CFP128)
 * @return The vector type name (e.g. "__daisy_type_complex_float")
 * @throws InvalidSDFGException if prim_type is not a complex type
 */
std::string complex_type_name(types::PrimitiveType prim_type);

/**
 * @brief Generates the C-style computation for a complex-valued tasklet as a single string.
 *
 * The returned string is a complete, `;`-terminated sequence of assignment statements that
 * operate component-wise on the `.x` (real) / `.y` (imaginary) members of the operands and
 * write the result into the tasklet's output connector. Narrow component types (half/bfloat)
 * are widened to `float` for the arithmetic. This lowering is language-agnostic and shared by
 * all C-style language extensions.
 *
 * @param tasklet A tasklet whose code is a complex operation (is_complex(code) == true)
 * @param function The function context, used to resolve operand types
 * @return The computation as a single string (e.g. "_out.x = (float)a.x + (float)b.x; ...")
 * @throws InvalidSDFGException if the tasklet code is not a complex operation
 */
std::string complex_computation(const data_flow::Tasklet& tasklet, const Function& function);

/**
 * @brief Produces the complex-type support preamble (2-component vector type definitions).
 *
 * Defines the `__daisy_type_complex_*` structs used to represent complex scalars. Arithmetic
 * on these values is emitted inline by the dataflow dispatcher, so no helper functions are
 * generated here.
 *
 * @param device If true, the half-precision component uses the GPU element type (__fp16).
 * @return The preamble source string
 */
std::string complex_support_preamble(bool device);

} // namespace codegen
} // namespace sdfg
