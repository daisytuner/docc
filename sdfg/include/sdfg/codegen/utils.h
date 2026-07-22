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
 * Reuses the CUDA/HIP vector-type convention (e.g. float2/double2). The corresponding
 * types are provided natively on GPU targets and defined by complex_support_preamble()
 * on CPU targets. Members `.x` (real) and `.y` (imaginary) are used consistently.
 *
 * @param prim_type A complex primitive type (CHalf, CBFloat, CFloat, CDouble, CFP128)
 * @return The vector type name (e.g. "float2")
 * @throws InvalidSDFGException if prim_type is not a complex type
 */
std::string complex_type_name(types::PrimitiveType prim_type);

/**
 * @brief Maps a complex primitive type to the helper-function suffix used in codegen.
 * @param prim_type A complex primitive type
 * @return The suffix ("h", "bf", "f", "d" or "q")
 * @throws InvalidSDFGException if prim_type is not a complex type
 */
std::string complex_op_suffix(types::PrimitiveType prim_type);

/**
 * @brief Builds the C expression for a complex tasklet operation.
 *
 * Emits a call to the corresponding generated helper function, e.g.
 * "__daisy_cmul_f(_in1, _in2)". The element suffix is derived from the operand type.
 *
 * @param function Function context (for resolving connector types)
 * @param tasklet The complex tasklet
 * @return The C expression
 * @throws InvalidSDFGException if the tasklet code is not a complex operation
 */
std::string complex_tasklet(sdfg::Function& function, const data_flow::Tasklet& tasklet);

/**
 * @brief Produces the complex-type support preamble (type definitions + helper functions).
 *
 * The preamble defines the `__daisy_c*` helper functions used by complex tasklets. On CPU
 * targets the 2-component vector types are also defined; on GPU targets the native vector
 * types (float2/double2/half2/__nv_bfloat162) are reused.
 *
 * @param device If true, helper functions are annotated for host+device execution and the
 *               native GPU vector types are reused instead of being redefined.
 * @return The preamble source string
 */
std::string complex_support_preamble(bool device);

} // namespace codegen
} // namespace sdfg
